// Realtime voice assistant: OpenAI Realtime API over WebRTC, with function tools that drive the planet.

export interface VoiceTools {
  lookupFeature(query: string): unknown;
  goToFeature(query: string): unknown;
  setTime(year: number, month?: number): unknown;
  setPlayback(playing: boolean): unknown;
  getState(): unknown;
}

export interface VoiceSession {
  stop(): void;
}

const MODEL = 'gpt-realtime';

const TOOLS = [
  { type: 'function', name: 'lookup_feature', description: 'Look up a web platform feature from caniuse by name. Returns when each browser first shipped it and current support. Use this to answer questions such as "when did X ship".', parameters: { type: 'object', properties: { query: { type: 'string', description: 'Feature name, e.g. "ArrayBuffer", "ES6 classes", "CSS grid"' } }, required: ['query'] } },
  { type: 'function', name: 'go_to_feature', description: 'Rotate the planet to show a feature and highlight it. Also returns the feature facts.', parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } },
  { type: 'function', name: 'set_time', description: 'Move the timeline to a year (and optional month). Pauses playback.', parameters: { type: 'object', properties: { year: { type: 'integer' }, month: { type: 'integer', description: '1-12' } }, required: ['year'] } },
  { type: 'function', name: 'set_playback', description: 'Start or pause the automatic playback of the timeline.', parameters: { type: 'object', properties: { playing: { type: 'boolean' } }, required: ['playing'] } },
  { type: 'function', name: 'get_state', description: 'Current date of the visualisation and summary statistics.', parameters: { type: 'object', properties: {} } },
];

const INSTRUCTIONS = `You are the voice of "Web Platform Planet", a 3D visualisation where every mountain is a web platform feature from caniuse.com and its height is the share of browsers supporting it over time (2000 to today).
Answer questions about web features using the tools; never guess dates. Drive the visualisation when asked (go to a feature, jump to a year, play or pause).
Be brief and conversational: one or two sentences. Reply in the language the user speaks. When you navigate somewhere, say what you did in a few words.`;

export async function startVoice(
  apiKey: string,
  tools: VoiceTools,
  onStatus: (s: string) => void,
  onCaption: (who: 'you' | 'planet', text: string) => void,
): Promise<VoiceSession> {
  onStatus('Connecting…');
  // Prefer an ephemeral client secret; fall back to the key itself.
  let token = apiKey;
  try {
    const r = await fetch('https://api.openai.com/v1/realtime/client_secrets', {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: { type: 'realtime', model: MODEL } }),
    });
    if (r.ok) token = (await r.json()).value ?? apiKey;
  } catch (e) { console.warn('[voice] no ephemeral key', e); }

  const pc = new RTCPeerConnection();
  const audio = new Audio();
  audio.autoplay = true;
  pc.ontrack = (e) => { audio.srcObject = e.streams[0]; };
  const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
  pc.addTrack(mic.getTracks()[0], mic);
  const dc = pc.createDataChannel('oai-events');

  const send = (msg: unknown) => dc.send(JSON.stringify(msg));
  dc.onopen = () => {
    send({
      type: 'session.update',
      session: {
        type: 'realtime',
        instructions: INSTRUCTIONS,
        tools: TOOLS,
        tool_choice: 'auto',
        audio: {
          input: { transcription: { model: 'gpt-4o-mini-transcribe' }, turn_detection: { type: 'semantic_vad' } },
          output: { voice: 'marin' },
        },
      },
    });
    onStatus('Listening · say something');
  };
  dc.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    switch (ev.type) {
      case 'error': console.error('[voice]', ev); onStatus(`Error: ${ev.error?.message ?? 'unknown'}`); break;
      case 'conversation.item.input_audio_transcription.completed': onCaption('you', ev.transcript); break;
      case 'response.output_audio_transcript.done':
      case 'response.audio_transcript.done': onCaption('planet', ev.transcript); break;
      case 'response.function_call_arguments.done': {
        let args: Record<string, unknown> = {};
        try { args = JSON.parse(ev.arguments || '{}'); } catch { /* ignore */ }
        let result: unknown;
        try {
          switch (ev.name) {
            case 'lookup_feature': result = tools.lookupFeature(String(args.query ?? '')); break;
            case 'go_to_feature': result = tools.goToFeature(String(args.query ?? '')); break;
            case 'set_time': result = tools.setTime(Number(args.year), args.month === undefined ? undefined : Number(args.month)); break;
            case 'set_playback': result = tools.setPlayback(Boolean(args.playing)); break;
            case 'get_state': result = tools.getState(); break;
            default: result = { error: `unknown tool ${ev.name}` };
          }
        } catch (err) { result = { error: String(err) }; }
        send({ type: 'conversation.item.create', item: { type: 'function_call_output', call_id: ev.call_id, output: JSON.stringify(result) } });
        send({ type: 'response.create' });
        break;
      }
    }
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  const res = await fetch(`https://api.openai.com/v1/realtime/calls?model=${MODEL}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/sdp' },
    body: offer.sdp,
  });
  if (!res.ok) throw new Error(`Realtime API ${res.status}: ${(await res.text()).slice(0, 200)}`);
  await pc.setRemoteDescription({ type: 'answer', sdp: await res.text() });

  return {
    stop() {
      mic.getTracks().forEach((t) => t.stop());
      dc.close();
      pc.close();
      audio.srcObject = null;
      onStatus('');
    },
  };
}
