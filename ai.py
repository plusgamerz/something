import os

from flask import Flask, jsonify, request, render_template_string
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = Flask(__name__)

MODEL_NAME = os.environ.get('LOCAL_AI_MODEL', 'google/flan-t5-xl')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_INPUT_TOKENS = 512  # guard against silent truncation on long histories

# Load model once at startup.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
except Exception as exc:
    raise RuntimeError(f'Failed to load local model {MODEL_NAME}: {exc}')

# Single source of truth for the system prompt — not duplicated in the frontend.
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
  <title>Nova AI Chatbot</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f4f7fb; }
    .chat-container { max-width: 800px; margin: 0 auto; padding: 24px; }
    h1 { margin-bottom: 16px; }
    .messages { list-style: none; padding: 0; margin: 0 0 16px; }
    .message { margin-bottom: 12px; padding: 14px 16px; border-radius: 12px; max-width: 80%; word-wrap: break-word; }
    .message.user { background: #fff; border: 1px solid #d7dde9; margin-left: auto; text-align: right; }
    .message.bot { background: #0b63ff; color: #fff; text-align: left; }
    .message.error { background: #ff4444; color: #fff; text-align: left; }
    .input-row { display: flex; gap: 8px; }
    input[type="text"] { flex: 1; padding: 12px 14px; border-radius: 10px; border: 1px solid #ccd4df; font-size: 16px; }
    button { padding: 12px 18px; border: none; border-radius: 10px; background: #0b63ff; color: #fff; font-size: 16px; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .typing { font-style: italic; opacity: 0.7; }
  </style>
</head>
<body>
  <div class="chat-container">
    <h1>Nova AI Chatbot</h1>
    <ul class="messages" id="messages"></ul>
    <div class="input-row">
      <input id="prompt" type="text" placeholder="Ask something..." autocomplete="off" />
      <button id="send">Send</button>
    </div>
  </div>

  <script>
    const messagesEl = document.getElementById('messages');
    const promptInput = document.getElementById('prompt');
    const sendButton = document.getElementById('send');

    // Only user/assistant turns — system prompt lives on the server.
    let history = [];

    function addMessage(text, role) {
      const li = document.createElement('li');
      li.className = 'message ' + role;
      li.textContent = text;
      messagesEl.appendChild(li);
      messagesEl.scrollIntoView({ block: 'end' });
      return li;
    }

    async function sendMessage() {
      const prompt = promptInput.value.trim();
      if (!prompt) return;

      addMessage(prompt, 'user');
      history.push({ role: 'user', content: prompt });
      promptInput.value = '';
      sendButton.disabled = true;

      const typingEl = addMessage('Nova is typing…', 'bot typing');

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: history })
        });

        const data = await response.json();
        typingEl.remove();

        if (!response.ok) {
          addMessage(data.error || 'Server error', 'error');
        } else {
          const content = data.reply;
          history.push({ role: 'assistant', content });
          addMessage(content, 'bot');
        }
      } catch (err) {
        typingEl.remove();
        addMessage('Unable to reach server.', 'error');
      }

      sendButton.disabled = false;
      promptInput.focus();
    }

    sendButton.addEventListener('click', sendMessage);
    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); sendMessage(); }
    });
  </script>
</body>
</html>
'''


def build_prompt(messages: list[dict]) -> str:
    """
    Build a single string prompt from the conversation history.
    The system prompt is injected here on the server — the frontend
    should NOT send a system-role message to avoid duplication.
    """
    parts = [SYSTEM_PROMPT, '']
    for item in messages:
        role = item.get('role', '')
        content = item.get('content', '').strip()
        if not content or role == 'system':
            # Skip blank turns and any stray system messages from the client.
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

    # Warn rather than silently mangle if the conversation is too long.
    token_count = len(tokenizer.encode(prompt))
    if token_count > MAX_INPUT_TOKENS:
        # Trim the oldest non-system turns by re-building with a shorter window.
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
                # Prevent the model from immediately ending the response.
                min_new_tokens=4,
                # Discourage repetition.
                repetition_penalty=1.3,
            )

        # flan-t5 is an encoder-decoder: generated_ids contains ONLY the
        # decoder output (no prompt echo), so decode directly.
        reply = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        if not reply:
            reply = "I'm not sure how to answer that. Could you rephrase?"

        return jsonify({'reply': reply})

    except Exception as exc:
        app.logger.exception('Generation failed')
        return jsonify({'error': f'Generation error: {exc}'}), 500


if __name__ == '__main__':
    # Set debug=False for production; use an env var to toggle.
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug)