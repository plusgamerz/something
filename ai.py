import os

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)  # Fix: allow cross-origin requests so the page works from any origin

MODEL_NAME = 'google/flan-t5-xl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_INPUT_TOKENS = 512

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
except Exception as exc:
    raise RuntimeError(f'Failed to load local model {MODEL_NAME}: {exc}')

SYSTEM_PROMPT = (
    'You are Nova, a friendly local AI chatbot running entirely on this machine. '
    'Your name is Nova. If the user asks your name, respond with "My name is Nova." '
    'You are an informative conversational assistant: give clear, coherent, and helpful answers, '
    'and expand when the user requests more detail. '
    'When the user greets you, reply with a single short greeting and do not echo the greeting back verbatim. '
    'Do not mention external APIs, cloud services, or that you are a demo. '
    'Keep responses polite and focused; prefer clarity over verbosity.'
)

HTML_PAGE = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nova</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0d0f14;
      --surface: #161920;
      --surface-2: #1e2130;
      --border: #2a2f45;
      --accent: #6c63ff;
      --accent-dim: #3d38a0;
      --user-bubble: #1e2130;
      --nova-bubble: #6c63ff;
      --text: #e8eaf0;
      --text-muted: #7b82a0;
      --danger: #ff5c5c;
      --radius: 18px;
      --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    body {
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      height: 100dvh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ── Header ── */
    header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px 20px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      z-index: 10;
    }

    .avatar {
      width: 36px; height: 36px;
      background: var(--accent);
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px;
      flex-shrink: 0;
    }

    .header-info h1 {
      font-size: 15px;
      font-weight: 600;
      letter-spacing: 0.02em;
    }

    .status {
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 5px;
    }

    .status-dot {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: #3dd68c;
      display: inline-block;
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }

    /* ── Messages ── */
    #messages-wrap {
      flex: 1;
      overflow-y: auto;
      padding: 24px 16px 8px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      scroll-behavior: smooth;
    }

    #messages-wrap::-webkit-scrollbar { width: 4px; }
    #messages-wrap::-webkit-scrollbar-track { background: transparent; }
    #messages-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    .msg-row {
      display: flex;
      gap: 10px;
      max-width: 760px;
      width: 100%;
    }

    .msg-row.user { margin-left: auto; flex-direction: row-reverse; }
    .msg-row.bot  { margin-right: auto; }

    .msg-avatar {
      width: 28px; height: 28px;
      border-radius: 50%;
      background: var(--accent);
      display: flex; align-items: center; justify-content: center;
      font-size: 13px;
      flex-shrink: 0;
      margin-top: 2px;
    }

    .msg-row.user .msg-avatar { background: var(--surface-2); }

    .bubble {
      padding: 11px 15px;
      border-radius: var(--radius);
      line-height: 1.55;
      font-size: 14.5px;
      max-width: calc(100% - 44px);
      word-break: break-word;
    }

    .msg-row.user .bubble {
      background: var(--user-bubble);
      border: 1px solid var(--border);
      border-bottom-right-radius: 4px;
    }

    .msg-row.bot .bubble {
      background: var(--nova-bubble);
      color: #fff;
      border-bottom-left-radius: 4px;
    }

    .msg-row.error .bubble {
      background: var(--danger);
      color: #fff;
      border-bottom-left-radius: 4px;
    }

    /* typing dots */
    .typing-dots span {
      display: inline-block;
      width: 6px; height: 6px;
      background: rgba(255,255,255,0.7);
      border-radius: 50%;
      margin: 0 2px;
      animation: blink 1.2s infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink {
      0%, 80%, 100% { opacity: 0.2; transform: translateY(0); }
      40% { opacity: 1; transform: translateY(-4px); }
    }

    /* Welcome card */
    .welcome {
      text-align: center;
      margin: auto;
      padding: 32px 20px;
      opacity: 0.7;
    }
    .welcome .big { font-size: 40px; margin-bottom: 10px; }
    .welcome h2 { font-size: 20px; font-weight: 600; margin-bottom: 6px; }
    .welcome p { font-size: 14px; color: var(--text-muted); }

    /* ── Input bar ── */
    #input-bar {
      flex-shrink: 0;
      padding: 12px 16px 16px;
      background: var(--bg);
      border-top: 1px solid var(--border);
    }

    .input-inner {
      max-width: 760px;
      margin: 0 auto;
      display: flex;
      align-items: flex-end;
      gap: 10px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 8px 8px 8px 18px;
      transition: border-color 0.2s;
    }

    .input-inner:focus-within {
      border-color: var(--accent);
    }

    #prompt {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text);
      font-size: 15px;
      font-family: var(--font);
      resize: none;
      max-height: 120px;
      line-height: 1.5;
      padding: 4px 0;
    }

    #prompt::placeholder { color: var(--text-muted); }

    #send {
      width: 38px; height: 38px;
      border-radius: 50%;
      border: none;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
      transition: background 0.2s, transform 0.1s;
    }

    #send:hover:not(:disabled) { background: #7c74ff; }
    #send:active:not(:disabled) { transform: scale(0.92); }
    #send:disabled { background: var(--accent-dim); cursor: not-allowed; }

    #send svg { pointer-events: none; }

    .hint {
      text-align: center;
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 8px;
    }
  </style>
</head>
<body>

<header>
  <div class="avatar">✦</div>
  <div class="header-info">
    <h1>Nova</h1>
    <span class="status"><span class="status-dot"></span> Local · On-device</span>
  </div>
</header>

<div id="messages-wrap">
  <div class="welcome" id="welcome">
    <div class="big">✦</div>
    <h2>Hi, I'm Nova</h2>
    <p>Your private, on-device AI assistant.<br>Ask me anything.</p>
  </div>
</div>

<div id="input-bar">
  <div class="input-inner">
    <textarea id="prompt" rows="1" placeholder="Message Nova…"></textarea>
    <button id="send" title="Send">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <line x1="22" y1="2" x2="11" y2="13"></line>
        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
      </svg>
    </button>
  </div>
  <p class="hint">Running locally · your data never leaves this machine</p>
</div>

<script>
  const wrap = document.getElementById('messages-wrap');
  const promptEl = document.getElementById('prompt');
  const sendBtn = document.getElementById('send');
  const welcome = document.getElementById('welcome');

  let history = [];

  // Auto-resize textarea
  promptEl.addEventListener('input', () => {
    promptEl.style.height = 'auto';
    promptEl.style.height = Math.min(promptEl.scrollHeight, 120) + 'px';
  });

  function scrollBottom() {
    wrap.scrollTop = wrap.scrollHeight;
  }

  function addRow(role) {
    if (welcome) welcome.remove();

    const row = document.createElement('div');
    row.className = 'msg-row ' + role;

    const av = document.createElement('div');
    av.className = 'msg-avatar';
    av.textContent = role === 'user' ? '🧑' : '✦';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    row.appendChild(av);
    row.appendChild(bubble);
    wrap.appendChild(row);
    scrollBottom();
    return bubble;
  }

  async function sendMessage() {
    const text = promptEl.value.trim();
    if (!text) return;

    addRow('user').textContent = text;
    history.push({ role: 'user', content: text });

    promptEl.value = '';
    promptEl.style.height = 'auto';
    sendBtn.disabled = true;

    // Typing indicator
    const typingBubble = addRow('bot');
    typingBubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history })
      });

      const data = await res.json();
      typingBubble.closest('.msg-row').remove();

      if (!res.ok) {
        addRow('error').textContent = data.error || 'Server error.';
      } else {
        history.push({ role: 'assistant', content: data.reply });
        addRow('bot').textContent = data.reply;
      }
    } catch (err) {
      typingBubble.closest('.msg-row').remove();
      const errBubble = addRow('error');
      errBubble.innerHTML =
        '<strong>Unable to reach Nova.</strong><br>' +
        'Make sure the server is running on <code style="background:rgba(0,0,0,.3);padding:1px 5px;border-radius:4px">http://localhost:5000</code> ' +
        'and you are accessing this page from that address.';
    }

    sendBtn.disabled = false;
    promptEl.focus();
    scrollBottom();
  }

  sendBtn.addEventListener('click', sendMessage);
  promptEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
</script>

</body>
</html>
'''


def build_prompt(messages: list[dict]) -> str:
    parts = [SYSTEM_PROMPT, '']
    for item in messages:
        role = item.get('role', '')
        content = item.get('content', '').strip()
        if not content or role == 'system':
            continue
        if role == 'user':
            parts.append(f'User: {content}')
        elif role == 'assistant':
            parts.append(f'Nova: {content}')
    parts.append('Nova:')
    return '\n'.join(parts)


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True)
    if not data or 'messages' not in data or not isinstance(data['messages'], list):
        return jsonify({'error': 'Invalid request: expected {"messages": [...]}'}), 400

    prompt = build_prompt(data['messages'])

    token_count = len(tokenizer.encode(prompt))
    if token_count > MAX_INPUT_TOKENS:
        msgs = data['messages']
        while len(msgs) > 1:
            msgs = msgs[1:]
            prompt = build_prompt(msgs)
            if len(tokenizer.encode(prompt)) <= MAX_INPUT_TOKENS:
                break

    try:
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                min_new_tokens=4,
                repetition_penalty=1.3,
            )

        reply = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        # Strip any "Nova:" prefix the model echoes back
        if reply.lower().startswith('nova:'):
            reply = reply[len('nova:'):].strip()

        if not reply:
            reply = "I'm not sure how to answer that. Could you rephrase?"

        return jsonify({'reply': reply})

    except Exception as exc:
        app.logger.exception('Generation failed')
        return jsonify({'error': f'Generation error: {exc}'}), 500


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    # Changed default port to 5001 to avoid macOS AirPlay conflict on port 5000
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=debug)