// Whisper browser proof of concept. Model downloads once; browser cache availability varies.
let asrBusy=false,asrCancel=false,asrPipeline=null;
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
  if(!asrPipeline){
   const lib=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1');
   lib.env.allowLocalModels=false;
   asrPipeline=await lib.pipeline('automatic-speech-recognition','onnx-community/whisper-tiny',{device:'wasm',dtype:'q8',progress_callback:p=>{if(p.status==='progress')asrSay('Téléchargement du modèle : '+Math.round(p.progress||0)+' %')}});
  }
  const sr=16000,frame=25*sr;
  const duration=decoded.duration;
  // Downmix via OfflineAudioContext to 16 kHz; bounded to ten minutes in this version.
  asrSay('Préparation audio 16 kHz…');
  const offline=new OfflineAudioContext(1,Math.ceil(duration*sr),sr);
  const src=offline.createBufferSource();src.buffer=decoded;
  src.connect(offline.destination);src.start(0);
  const mono=(await offline.startRendering()).getChannelData(0);
  const total=Math.ceil(mono.length/frame);
  let textSoFar=meeting.transcript||'';
  for(let i=0;i<total;i++){
   if(asrCancel)break;
   asrSay('Transcription locale : segment '+(i+1)+' / '+total+'. Gardez Safari ouvert.');
   const segment=mono.slice(i*frame,Math.min((i+1)*frame,mono.length));
   const result=await asrPipeline(segment,{language:'french',task:'transcribe'});
   const words=(result.text||'').trim();
   if(words){
    textSoFar+=(textSoFar?'\n':'')+'['+Math.floor(i*25/60).toString().padStart(2,'0')+':'+String(i*25%60).padStart(2,'0')+'] '+words;
    const latest=await get(meetingId);latest.transcript=textSoFar;await save(latest);
    if(current===meetingId)document.getElementById('transcript').value=textSoFar;
   }
  }
  asrSay(asrCancel?'Arrêt effectué. Les segments terminés sont enregistrés.':'Transcription terminée et enregistrée.');
 }catch(e){asrSay('Échec : '+(e?.message||String(e))+' — modèle ou mémoire potentiellement incompatibles avec Safari.')}
 finally{if(ctx)await ctx.close().catch(()=>{});asrBusy=false;asrButton.disabled=false;asrStop.disabled=true}
});
