// Whisper browser proof of concept. Model downloads once; browser cache availability varies.
let asrBusy=false,asrCancel=false,asrPipeline=null,asrLoadedModel=null;
const asrButton=document.getElementById('transcribe');
const asrStop=document.getElementById('cancelTranscribe');
const asrMessage=document.getElementById('asrStatus');
const asrSay=(s)=>{asrMessage.textContent=s};
asrStop.addEventListener('click',()=>{asrCancel=true;asrSay('Arrêt demandé : fin du segment en cours…')});
asrButton.addEventListener('click',async()=>{
 if(asrBusy||!current)return;
 asrBusy=true;asrCancel=false;asrButton.disabled=true;asrStop.disabled=false;
 let ctx;
 try{
  const meetingId=current;
  const meeting=await get(meetingId);
  if(!meeting?.audio)throw Error('Enregistrement introuvable.');
  asrSay('Lecture et décodage du fichier audio…');
  ctx=new (window.AudioContext||window.webkitAudioContext)();
  const bytes=await meeting.audio.arrayBuffer();
  const decoded=await ctx.decodeAudioData(bytes);
  if(decoded.duration>600)throw Error('Cette version expérimentale accepte au maximum 10 minutes. Pour une réunion d’une heure, conservez l’audio et attendez la version optimisée.');
  asrSay('Chargement du moteur Whisper : le premier téléchargement peut être long…');
  const modelChoice=document.getElementById('asrModel').value;
  const languageChoice=document.getElementById('asrLang').value;
  if(modelChoice==='large-v3-turbo' && !confirm('Large V3 Turbo est très volumineux et peut provoquer le rechargement de Safari sur iPhone. Les résultats ne sont pas garantis. Continuer ?')){asrSay('Test haute qualité annulé.');return;}
  if(!asrPipeline||asrLoadedModel!==modelChoice){
   asrPipeline=null;
   const lib=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1');
   lib.env.allowLocalModels=false;
   const highQuality=modelChoice==='large-v3-turbo';
   const device=highQuality && navigator.gpu?'webgpu':'wasm';
   const dtype=highQuality && device==='webgpu'?{encoder_model:'fp16',decoder_model_merged:'q4'}:'q8';
   asrPipeline=await lib.pipeline('automatic-speech-recognition','onnx-community/whisper-'+modelChoice,{device,dtype,progress_callback:p=>{if(p.status==='progress')asrSay('Téléchargement du modèle : '+Math.round(p.progress||0)+' %')}});
   asrLoadedModel=modelChoice;
  }
  const sr=16000,frame=25*sr;
  const duration=decoded.duration;
  // Downmix via OfflineAudioContext to 16 kHz; bounded to ten minutes in this version.
  asrSay('Préparation audio 16 kHz…');
  // Explicit mono mixing and resampling avoids Safari OfflineAudioContext inconsistencies.
  const sourceRate=decoded.sampleRate;
  const sourceLength=decoded.length;
  const outputLength=Math.round(sourceLength*sr/sourceRate);
  const mono=new Float32Array(outputLength);
  const channels=decoded.numberOfChannels;
  const data=Array.from({length:channels},(_,i)=>decoded.getChannelData(i));
  for(let i=0;i<outputLength;i++){
   const pos=i*sourceRate/sr,lo=Math.min(sourceLength-1,Math.floor(pos)),hi=Math.min(sourceLength-1,lo+1),fraction=pos-lo;
   let sample=0;
   for(let c=0;c<channels;c++)sample+=data[c][lo]*(1-fraction)+data[c][hi]*fraction;
   mono[i]=sample/channels;
  }
  let energy=0,peak=0;
  for(let i=0;i<mono.length;i++){energy+=mono[i]*mono[i];peak=Math.max(peak,Math.abs(mono[i]));}
  const rms=Math.sqrt(energy/Math.max(1,mono.length));
  if(rms<0.0001)throw Error('Signal silencieux : transcription impossible.');
  // Normalize quiet speech to approximately -20 dBFS, without clipping.
  const gain=Math.min(0.85/Math.max(peak,0.000001),0.1/Math.max(rms,0.000001),30);
  if(gain>1.01){for(let i=0;i<mono.length;i++)mono[i]*=gain;}
  asrSay('Niveau audio original : '+Math.round(20*Math.log10(rms))+' dBFS. Gain appliqué : '+gain.toFixed(1)+'×.');
  asrSay('Signal vérifié : '+sourceRate+' Hz → 16000 Hz, '+channels+' canal(aux), niveau '+Math.round(20*Math.log10(rms))+' dBFS.');
  const total=Math.ceil(mono.length/frame);
  let textSoFar='';
  if(meeting.transcript && !confirm('Remplacer la transcription existante par une nouvelle transcription ?')){asrSay('Transcription annulée.');return;}
  for(let i=0;i<total;i++){
   if(asrCancel)break;
   asrSay('Transcription locale : segment '+(i+1)+' / '+total+'. Gardez Safari ouvert.');
   const segment=mono.slice(i*frame,Math.min((i+1)*frame,mono.length));
   const options={task:'transcribe'};
   if(languageChoice!=='auto')options.language=languageChoice;
   options.condition_on_prev_tokens=false;
   options.no_repeat_ngram_size=4;
   options.repetition_penalty=1.15;
   options.max_new_tokens=128;
   const result=await asrPipeline(segment,options);
   const words=(result.text||'').trim();
   const wordCount=words.split(/\\s+/).filter(Boolean).length;
   const seconds=segment.length/sr;
   if(wordCount>seconds*3.5+10){asrSay('Résultat incohérent : trop de mots pour la durée. Aucun texte enregistré pour ce segment.');continue;}
   const phrases=words.split(/[.!?]+/).map(x=>x.trim().toLowerCase()).filter(Boolean);
   const repeated=phrases.some(p=>phrases.filter(q=>q===p).length>=3);
   if(repeated){asrSay('Transcription suspecte : répétitions détectées. Résultat non enregistré.');continue;}
   if(words){
    textSoFar+=(textSoFar?'\n':'')+'['+Math.floor(i*25/60).toString().padStart(2,'0')+':'+String(i*25%60).padStart(2,'0')+'] '+words;
    const latest=await get(meetingId);latest.transcript=textSoFar;await save(latest);
    if(current===meetingId)document.getElementById('transcript').value=textSoFar;
   }
  }
  asrSay(asrCancel?'Arrêt demandé.':textSoFar?'Transcription enregistrée (résultat expérimental à vérifier).':'Aucun texte fiable obtenu. Vérifiez le fichier audio et essayez un autre modèle.');
 }catch(e){asrSay('Échec : '+(e?.message||String(e))+' — modèle ou mémoire potentiellement incompatibles avec Safari.')}
 finally{if(ctx)await ctx.close().catch(()=>{});asrBusy=false;asrButton.disabled=false;asrStop.disabled=true}
});
