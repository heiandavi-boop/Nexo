import {ChunkSender} from './vad.js';
const $=id=>document.getElementById(id);
let context,stream,node,source,muted,playing,sender,workletContext,permissionStatus;
let enabled=false,busy=false,generation=0,pendingStart=0,startSeq=0;
let needsGesture=false,audioState='sin activar',micPermission='desconocido',autoTried=false;
let faceFrame=0;
const SETTINGS={umbral_inicio:'sensitivity',pausa_corta:'shortPause',pausa_larga:'longPause'};
const PERMISSION_NAMES={granted:'concedido',prompt:'sin decidir',denied:'denegado'};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const withTimeout=(promise,ms,message)=>Promise.race([promise,new Promise((_,reject)=>setTimeout(()=>{const e=Error(message);e.name='TimeoutError';reject(e);},ms))]);
// Salida independiente del AudioContext de captura: cambiar de micrófono no puede silenciar la voz.
const voice=new Audio();
function envelope(bytes){
  const view=new DataView(bytes.buffer);let p=12,rate=22050,channels=1,start=0,size=0;
  while(p+8<=view.byteLength){
    const id=String.fromCharCode(...bytes.subarray(p,p+4)),len=view.getUint32(p+4,true);
    if(id==='fmt '){channels=view.getUint16(p+10,true);rate=view.getUint32(p+12,true);}
    if(id==='data'){start=p+8;size=Math.min(len,view.byteLength-start);break;}
    p+=8+len+(len&1);
  }
  const step=Math.max(1,Math.round(rate*.02))*channels*2,levels=[];
  for(let i=start;i+1<start+size;i+=step){let sum=0,n=0;for(let j=i;j+1<Math.min(i+step,start+size);j+=2){const x=view.getInt16(j,true)/32768;sum+=x*x;n++;}levels.push(Math.sqrt(sum/(n||1)));}
  return levels;
}
function animateVoice(levels){
  cancelAnimationFrame(faceFrame);
  const frame=()=>{
    const rms=levels[Math.floor(voice.currentTime/.02)]||0;
    $('orb').style.setProperty('--mouth-open', Math.min(1,rms*7).toFixed(3));
    faceFrame=requestAnimationFrame(frame);
  };
  frame();
}
// Reproduce un silencio para comprobar si el navegador permite sonido; dentro de un clic también lo habilita.
function primeVoice(){
  if(playing)return Promise.resolve(true);
  const silent=new Uint8Array(46),view=new DataView(silent.buffer),text=(p,s)=>[...s].forEach((c,i)=>silent[p+i]=c.charCodeAt(0));
  text(0,'RIFF');view.setUint32(4,38,true);text(8,'WAVEfmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,8000,true);view.setUint32(28,16000,true);view.setUint16(32,2,true);view.setUint16(34,16,true);text(36,'data');view.setUint32(40,2,true);
  const url=URL.createObjectURL(new Blob([silent],{type:'audio/wav'}));voice.src=url;
  return voice.play().then(()=>{audioState='habilitada';return true;},e=>{if(e.name==='NotAllowedError')audioState='bloqueada';return audioState!=='bloqueada';}).finally(()=>{URL.revokeObjectURL(url);showAccess();});
}
function restFace(){cancelAnimationFrame(faceFrame);$('orb').style.setProperty('--mouth-open','0');}
const error=text=>{$('error').textContent=text;$('error').hidden=!text;};
function state(name,title,detail){$('orb').dataset.state=name;if(name!=='speaking')restFace();$('status').textContent=title;$('detail').textContent=detail;}
function listening(){return enabled&&!!sender?.open&&!!stream&&context?.state==='running';}
function idle(){
  if(needsGesture)return state('paused','Toca una vez para habilitar la voz','El navegador necesita una interacción para activar el micrófono y el audio.');
  if(pendingStart)return state('paused','Activando…','Preparando el micrófono y el audio en este Mac.');
  if(listening())return state('listening','Escuchando','Habla cuando quieras. Respondo cuando termines la frase.');
  state('paused',enabled?'Activando…':'Conversación en pausa',enabled?'Preparando la escucha.':'Activa la conversación cuando quieras.');
}
function refreshUI(){
  $('start').textContent=pendingStart?'Cancelar activación':enabled?'Pausar conversación':'Iniciar conversación ↗';
  const activate=$('activate');
  activate.hidden=!(needsGesture||(enabled&&audioState==='bloqueada'));
  activate.textContent=needsGesture?'Activar micrófono y audio':'Toca una vez para habilitar la voz';
  showAccess();
}
function showAccess(){
  const mic=stream?'en uso':'apagado';
  $('access').textContent=`Permiso del micrófono: ${micPermission} · Micrófono: ${mic} · Voz: ${audioState}`;
  $('access').dataset.mic=stream?'on':'off';
}
function count(n){$('count').textContent=`${n} ${n===1?'recuerdo':'recuerdos'}`;}
function profile(facts){$('profile').textContent=Object.entries(facts||{}).map(([k,v])=>`${k}: ${v}`).join('\n')||'Aún no hay datos personales confirmados.';}
function timings(t={}){const names={fin_de_turno:'fin de turno',transcripcion:'transcripción',generacion:'generación',sintesis_voz:'voz'};$('timings').textContent=Object.entries(names).filter(([k])=>k in t).map(([k,n])=>`${n} ${t[k].toFixed(2)} s`).join(' · ');}
function bubble(role,text,doubtful=[],note=''){
  $('empty')?.remove();const el=document.createElement('div');el.className=`bubble ${role}`;const who=document.createElement('span');who.className='who';who.textContent=role==='user'?'Tú':'Nexo';el.append(who);
  const unsure=new Set(doubtful.map(w=>w.toLowerCase()));
  for(const part of unsure.size?text.split(/(\s+)/):[text]){
    const bare=part.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu,'').toLowerCase();
    if(bare&&unsure.has(bare)){const mark=document.createElement('mark');mark.title='Reconocida con poca seguridad';mark.textContent=part;el.append(mark);}else el.append(document.createTextNode(part));
  }
  if(note){const small=document.createElement('small');small.className='bubble-note';small.textContent=note;el.append(small);}
  $('messages').append(el);$('messages').scrollTop=$('messages').scrollHeight;return el;
}
async function history(){const r=await fetch('/api/history');if(!r.ok)throw Error('No pude cargar la memoria.');const d=await r.json();$('messages').replaceChildren();for(const t of d.turns){bubble('user',t.question);bubble('assistant',t.answer);}count(d.count);profile(d.profile);if(!d.count){$('messages').innerHTML='<div class="empty" id="empty"><span>✳</span><h3>Todo empieza con un hola.</h3><p>Tu conversación se guarda en este Mac para continuar donde la dejaste.</p></div>';}}
async function checkStatus(){
  try{const r=await fetch('/api/status');const s=await r.json();$('engine').textContent=`Reconocimiento: ${s.voz.motor}`;
    const problems=[s.voz.ok?'':s.voz.mensaje,s.lmstudio.ok?'':s.lmstudio.mensaje].filter(Boolean);if(problems.length)error(problems.join(' '));return s;}
  catch{error('El servidor local no responde. Ejecuta .venv/bin/python servidor.py');}
}
async function api(url,body,timeout=0){
  const controller=new AbortController(),timer=timeout&&setTimeout(()=>controller.abort(),timeout);
  let r;
  try{r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Local-App':'1'},body,signal:controller.signal});}
  catch(e){throw Error(e.name==='AbortError'?'El servidor local no respondió a tiempo. Comprueba que siga abierto.':'Se perdió la conexión con el servidor local.');}
  finally{clearTimeout(timer);}
  let data={};try{data=await r.json();}catch{}
  if(!r.ok){const e=Error(data.error||`Error ${r.status} del servidor local.`);e.status=r.status;e.data=data;throw e;}
  return data;
}
async function postChunk(body,token){
  const r=await fetch('/api/voice/chunk',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Local-App':'1','X-Voice-Token':token},body});
  if(r.status===409)return null;
  const data=await r.json();if(!r.ok)throw Error(data.error||'Falló la escucha local.');return data;
}
function settings(){const out={};for(const [key,id] of Object.entries(SETTINGS))out[key]=Number($(id).value);return out;}
function showSettings(){for(const id of Object.values(SETTINGS))$(id+'Value').value=Number($(id).value).toFixed(id==='sensitivity'?2:1);}
const DENIED_HELP='El micrófono está bloqueado para http://localhost:8765. En la barra de dirección pulsa el icono del candado o del micrófono, elige «Permitir» y recarga la página. Revisa también Ajustes del Sistema → Privacidad y seguridad → Micrófono.';
function micError(e){
  let message=`No pude activar el micrófono: ${e.message||e.name}`;
  if(!navigator.mediaDevices?.getUserMedia)message='Este navegador no permite el micrófono aquí. Abre http://localhost:8765 en Safari o Chrome.';
  else if(['NotAllowedError','SecurityError'].includes(e.name))message=DENIED_HELP;
  else if(['NotFoundError','OverconstrainedError'].includes(e.name))message='No encontré ningún micrófono. Conecta uno o elígelo en Ajustes del Sistema → Sonido.';
  else if(e.name==='NotReadableError')message='El micrófono está ocupado por otra aplicación o macOS lo bloqueó.';
  const result=Error(message);result.name=e.name||'Error';return result;
}
async function queryPermission(){
  try{
    if(!permissionStatus){
      permissionStatus=await navigator.permissions.query({name:'microphone'});
      permissionStatus.onchange=()=>{
        micPermission=PERMISSION_NAMES[permissionStatus.state]||'desconocido';
        if(micPermission==='denegado'&&(enabled||pendingStart)){stop();error(DENIED_HELP);}
        refreshUI();
      };
    }
    micPermission=PERMISSION_NAMES[permissionStatus.state]||'desconocido';
  }catch{micPermission='desconocido';}
  showAccess();return micPermission;
}
function onVoice(e){
  $('level').style.width=`${Math.min(100,e.level*900)}%`;
  if(e.error){stop();error('No pude transcribir tu voz: '+e.error);return;}
  if(e.turn){sender.close();handleTurn(e.turn);return;}
  if(busy)return;
  if(e.state==='transcribiendo')state('transcribing','Transcribiendo…','Convirtiendo tu voz en texto en este Mac.');
  else if(e.state==='pausa'&&e.incomplete)state('listening','Escuchando','Parece que la frase continúa; te doy un poco más de tiempo.');
  else if(e.state==='voz'||e.state==='pausa')state('listening','Escuchando','Continúa; respondo cuando termines.');
  else if(e.discarded)state('listening','Escuchando',`Ignoré un sonido (${e.discarded}). Habla cuando quieras.`);
}
async function listen(){
  const ticket=generation;
  const data=await api('/api/voice/start',JSON.stringify(settings()),8000);
  if(ticket!==generation||!enabled)return;
  sender.start(data.token);idle();refreshUI();
}
// Nunca espera indefinidamente: si el navegador exige un gesto, resume() puede quedar pendiente.
async function resumeAudio(){
  if(!context||context.state==='closed'){
    context=new AudioContext();
    context.onstatechange=()=>{
      if(enabled&&context.state!=='running'&&context.state!=='closed'){needsGesture=true;sender?.close();}
      else if(enabled&&context.state==='running'&&needsGesture){needsGesture=false;if(!busy)listen().catch(e=>{stop();error(e.message);});}
      if(!busy)idle();refreshUI();
    };
  }
  if(context.state!=='running')await Promise.race([context.resume().catch(()=>{}),sleep(400)]);
  return context.state==='running';
}
async function refreshMics(){
  if(!navigator.mediaDevices?.enumerateDevices)return;
  const inputs=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='audioinput'&&d.deviceId&&!['default','communications'].includes(d.deviceId));
  const saved=localStorage.getItem('nexo.mic')||'',select=$('mic');
  select.replaceChildren(new Option('Predeterminado de macOS',''));
  inputs.forEach((d,i)=>select.append(new Option(d.label||`Micrófono ${i+1} (inicia la conversación para ver el nombre)`,d.deviceId)));
  select.value=inputs.some(d=>d.deviceId===saved)?saved:'';
}
function stopTracks(s){s?.getTracks().forEach(t=>{t.onended=null;t.stop();});}
function releaseCapture(){
  stopTracks(stream);stream=null;
  source?.disconnect();node?.disconnect();muted?.disconnect();
  if(node)node.port.onmessage=null;
  node=source=muted=null;
}
function stop(){
  const wasActive=enabled||pendingStart;
  enabled=false;pendingStart=0;startSeq++;generation++;needsGesture=false;sender?.close();
  if(playing){playing.stop();playing=null;}
  releaseCapture();
  if(wasActive)fetch('/api/voice/stop',{method:'POST',headers:{'X-Local-App':'1'}}).catch(()=>{});
  $('level').style.width='0%';$('micName').textContent='';idle();refreshUI();
}
// Con permiso ya concedido no hay pregunta pendiente: si el micrófono no responde, se avisa en vez de esperar sin fin.
async function requestStream(constraints){
  const request=navigator.mediaDevices.getUserMedia(constraints);
  if(micPermission!=='concedido')return request;
  let expired=false;
  request.then(s=>{if(expired)stopTracks(s);},()=>{});
  try{return await withTimeout(request,8000,'El micrófono no responde. Reconéctalo o elige otro en la lista y vuelve a intentarlo.');}
  catch(e){if(e.name==='TimeoutError')expired=true;throw e;}
}
async function getMicrophone(){
  const audio={channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true},chosen=$('mic').value;
  if(!navigator.mediaDevices?.getUserMedia)throw micError({name:'NotSupportedError'});
  try{return await requestStream({audio:chosen?{...audio,deviceId:{exact:chosen}}:audio,video:false});}
  catch(e){
    if(e.name==='TimeoutError')throw e;
    if(!chosen||!['OverconstrainedError','NotFoundError'].includes(e.name))throw micError(e);
    localStorage.removeItem('nexo.mic');
    try{const fallback=await requestStream({audio,video:false});error('No encontré el micrófono elegido; uso el predeterminado de macOS.');return fallback;}
    catch(x){throw micError(x);}
  }
}
async function start({voiceReady=null}={}){
  if(pendingStart||enabled)return;
  const seq=pendingStart=++startSeq;
  needsGesture=false;error('');idle();refreshUI();
  const stage=text=>{if(seq===startSeq)$('detail').textContent=text;};
  let captured=null;
  try{
    await queryPermission();
    stage('Abriendo el micrófono…');
    captured=await getMicrophone();
    // Si se pausó mientras el navegador pedía permiso, el stream que llega tarde se descarta.
    if(seq!==startSeq)return stopTracks(captured);
    await queryPermission();
    if(micPermission!=='concedido')micPermission='concedido en esta sesión';
    stage('Activando el audio…');
    const running=await resumeAudio();
    if(seq!==startSeq)return stopTracks(captured);
    if(!running){stopTracks(captured);needsGesture=true;return;}
    // Si el navegador no resuelve la prueba de sonido, se sigue: la voz avisará si queda bloqueada.
    if(!(await Promise.race([voiceReady||primeVoice(),sleep(800).then(()=>true)])))audioState='bloqueada';
    if(seq!==startSeq)return stopTracks(captured);
    stage('Preparando el procesamiento de voz…');
    if(workletContext!==context){await withTimeout(context.audioWorklet.addModule('/capture.js'),8000,'No pude cargar el procesador de audio. Recarga la página.');workletContext=context;}
    if(seq!==startSeq)return stopTracks(captured);
    releaseCapture();
    stream=captured;captured=null;
    refreshMics().catch(()=>{});
    $('micName').textContent=`Usando: ${stream.getAudioTracks()[0].label||'micrófono predeterminado'}`;
    node=new AudioWorkletNode(context,'capture');source=context.createMediaStreamSource(stream);muted=context.createGain();muted.gain.value=0;source.connect(node);node.connect(muted);muted.connect(context.destination);
    sender??=new ChunkSender(postChunk,onVoice,e=>{stop();error(e.message);});
    // Durante la respuesta no se envía audio: así la voz del asistente no se toma como un turno nuevo.
    node.port.onmessage=({data})=>{if(enabled&&!busy&&!document.hidden)sender.push(data);};
    stream.getAudioTracks()[0].onended=()=>{stop();error('El micrófono se desconectó. Vuelve a activar la conversación.');};
    enabled=true;generation++;
    stage('Conectando con el reconocimiento de voz…');
    await listen();
  }catch(e){
    stopTracks(captured);
    if(seq!==startSeq)return;
    enabled=false;releaseCapture();
    if(e.name==='NotAllowedError'&&(await queryPermission())!=='denegado'){
      needsGesture=true;error('El navegador no permitió activar el micrófono automáticamente. Pulsa «Activar micrófono y audio».');
    }else error(e.message);
    if(e.data?.model)checkStatus();
  }finally{
    if(pendingStart===seq)pendingStart=0;
    if(!busy)idle();refreshUI();
  }
}
async function play(base64,ticket,onStart){
  const bytes=Uint8Array.from(atob(base64),c=>c.charCodeAt(0)),levels=envelope(bytes);
  if(ticket!==generation)return;
  const url=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));
  try{
    await new Promise((resolve,reject)=>{
      playing={stop:()=>{voice.pause();resolve();}};
      voice.onended=resolve;voice.onerror=()=>reject(Error('No pude reproducir la voz de Nexo.'));
      voice.src=url;voice.volume=1;voice.muted=false;
      const stalled=setTimeout(()=>{voice.pause();reject(Error('La voz no empezó a sonar. La respuesta está en pantalla.'));},4000);
      voice.play().then(()=>{clearTimeout(stalled);audioState='habilitada';onStart?.();animateVoice(levels);showAccess();},e=>{
        clearTimeout(stalled);
        const failure=Error(e.name==='NotAllowedError'?'El navegador bloqueó la voz. Pulsa «Toca una vez para habilitar la voz»; la respuesta está en pantalla.':`No pude reproducir la voz: ${e.message}`);
        if(e.name==='NotAllowedError'){audioState='bloqueada';failure.blocked=true;}
        reject(failure);
      });
    });
  }finally{playing=null;voice.onended=voice.onerror=null;restFace();URL.revokeObjectURL(url);refreshUI();}
}
async function respond(payload,text,doubtful=[]){
  payload={...payload,voice:$('voiceSelect').value};
  busy=true;const ticket=generation;error('');$('send').disabled=true;$('clear').disabled=true;
  const mine=bubble('user',text,doubtful,doubtful.length?'Las palabras resaltadas se reconocieron con poca seguridad.':'');
  state('thinking','Pensando…','Qwen prepara la respuesta en este Mac.');
  try{
    const data=await api('/api/talk',JSON.stringify(payload));
    if(data.empty){mine.remove();error('No detecté palabras claras. Puedes intentarlo otra vez.');return;}
    bubble('assistant',data.answer,[],data.saved===false?'No guardé este intercambio: pedí que lo repitas.':'');count(data.count);profile(data.profile);timings(data.timings);
    if(data.warning)error(data.warning);
    if(data.audio&&ticket===generation)await play(data.audio,ticket,()=>state('speaking','Hablando','La escucha se reanuda al terminar mi respuesta.'));
  }catch(e){if(e.data?.duplicate)mine.remove();else error(e.message||'Se perdió la conexión con el servidor local.');}
  finally{
    busy=false;$('send').disabled=false;$('clear').disabled=false;
    if(enabled&&ticket===generation){
      // Margen para que el eco de la última sílaba del asistente se disipe.
      await sleep(350);
      if(enabled&&ticket===generation&&!needsGesture)listen().catch(e=>{stop();error(e.message);});
      else idle();
    }else if(!pendingStart)idle();
  }
}
function handleTurn(turn){
  if(busy)return;
  timings(turn.timings);
  respond({turn:turn.id},turn.text,turn.doubtful||[]);
}
// Se llama siempre dentro de un clic: así el navegador acepta reanudar el audio.
function userStart(){const voiceReady=primeVoice();resumeAudio();start({voiceReady});}
async function activate(){
  const voiceReady=primeVoice(),audioReady=resumeAudio();
  if(!enabled){needsGesture=false;return start({voiceReady});}
  await voiceReady;
  if(!(await audioReady))needsGesture=true;
  idle();refreshUI();
}
async function autoStart(){
  await queryPermission();
  if(!$('autoStart').checked||autoTried||enabled||pendingStart)return;
  if(micPermission==='denegado'){error(DENIED_HELP);return;}
  if(document.hidden){document.addEventListener('visibilitychange',autoStart,{once:true});return;}
  // Un solo intento automático por carga, nunca en bucle.
  autoTried=true;
  await start();
}
$('start').onclick=()=>enabled||pendingStart?stop():userStart();
$('activate').onclick=activate;
$('textForm').onsubmit=async e=>{e.preventDefault();if(busy)return;const text=$('text').value.trim();if(!text)return;primeVoice();resumeAudio();$('text').value='';sender?.close();await respond({text},text);};
for(const [key,id] of Object.entries(SETTINGS)){
  const saved=localStorage.getItem('nexo.'+key);if(saved!==null)$(id).value=saved;
  $(id).oninput=()=>{localStorage.setItem('nexo.'+key,$(id).value);showSettings();};
}
showSettings();
$('autoStart').checked=localStorage.getItem('nexo.autoinicio')!=='0';
$('autoStart').onchange=()=>localStorage.setItem('nexo.autoinicio',$('autoStart').checked?'1':'0');
async function loadVoices(){
  const r=await fetch('/api/voices');if(!r.ok)return;const d=await r.json(),select=$('voiceSelect'),saved=localStorage.getItem('nexo.voz');
  select.replaceChildren(...d.voces.map(v=>new Option(v.nombre,v.id)));
  select.value=d.voces.some(v=>v.id===saved)?saved:d.predeterminada;
}
$('voiceSelect').onchange=()=>localStorage.setItem('nexo.voz',$('voiceSelect').value);
$('voiceTest').onclick=async()=>{
  if(busy)return;
  const voiceReady=primeVoice();resumeAudio();busy=true;sender?.close();const ticket=generation;$('voiceTest').disabled=true;error('');
  try{await voiceReady;const d=await api('/api/voice/sample',JSON.stringify({voice:$('voiceSelect').value}));await play(d.audio,ticket,()=>state('speaking','Hablando','Muestra de la voz elegida.'));}
  catch(e){error(e.message);}
  finally{busy=false;$('voiceTest').disabled=false;if(enabled&&ticket===generation&&!needsGesture)listen().catch(e=>{stop();error(e.message);});else idle();}
};
$('mic').onchange=()=>{localStorage.setItem('nexo.mic',$('mic').value);if(enabled||pendingStart){stop();userStart();}};
navigator.mediaDevices?.addEventListener('devicechange',()=>refreshMics().then(()=>{if(!busy)$('detail').textContent='Detecté un cambio de micrófonos. Elige el que quieras en la lista.';}));
$('clear').onclick=async()=>{if(busy||!confirm('¿Borrar todas las conversaciones guardadas en este Mac?'))return;stop();try{const r=await fetch('/api/clear',{method:'POST',headers:{'X-Local-App':'1'}});if(!r.ok)throw Error('No se pudo borrar la memoria.');await history();}catch(e){error(e.message);}};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&(enabled||pendingStart))stop();});
window.addEventListener('pagehide',()=>{stop();context?.close().catch(()=>{});});
window.addEventListener('pageshow',e=>{if(e.persisted){autoTried=false;autoStart();}});
history().catch(e=>error(e.message));
checkStatus();
loadVoices().catch(()=>{});
refreshMics().catch(()=>{});
idle();refreshUI();
autoStart();
