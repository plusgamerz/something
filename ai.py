import os

from flask import Flask, jsonify, request, render_template_string
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = Flask(__name__)

MODEL_NAME = os.environ.get('LOCAL_AI_MODEL', 'google/flan-t5-xl')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model once at startup.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # avoid tied weights warning for some HF checkpoints
    try:
      model.config.tie_word_embeddings = False
    except Exception:
      pass
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.to(DEVICE)
    text_generator = None
except Exception as exc:
    raise RuntimeError(f'Failed to load local model {MODEL_NAME}: {exc}')

SYSTEM_PROMPT = (
  'You are Nova, a friendly local AI chatbot running entirely on this machine. '
  'Your name is Nova. If the user asks your name, respond with "My name is Nova." '
  'You are an informative conversational assistant: give clear, coherent, and helpful answers, and expand when the user requests more detail. '
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
    .messages { list-style: none; padding: 0; margin: 0 0 16px; }
    .message { margin-bottom: 12px; padding: 14px 16px; border-radius: 12px; }
    .message.user { background: #fff; border: 1px solid #d7dde9; text-align: right; }
    .message.bot { background: #0b63ff; color: #fff; text-align: left; }
    .input-row { display: flex; gap: 8px; }
    input[type="text"] { flex: 1; padding: 12px 14px; border-radius: 10px; border: 1px solid #ccd4df; font-size: 16px; }
    button { padding: 12px 18px; border: none; border-radius: 10px; background: #0b63ff; color: #fff; font-size: 16px; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="chat-container">
    <h1>AI Chatbot</h1>
    <ul class="messages" id="messages"></ul>
    <div class="input-row">
      <input id="prompt" type="text" placeholder="Ask something..." autocomplete="off" />
      <button id="send">Send</button>
    </div>
  </div>

  <script>
    const messages = document.getElementById('messages');
    const promptInput = document.getElementById('prompt');
    const sendButton = document.getElementById('send');

    let history = [
      { role: 'system', content: 'You are Nova, a local AI chatbot. Your name is Nova. Answer clearly and politely.' }
    ];

    function addMessage(text, role) {
      const li = document.createElement('li');
      li.className = 'message ' + role;
      li.textContent = text;
      messages.appendChild(li);
      messages.scrollTop = messages.scrollHeight;
    }

    async function sendMessage() {
      const prompt = promptInput.value.trim();
      if (!prompt) return;

      addMessage(prompt, 'user');
      history.push({ role: 'user', content: prompt });
      promptInput.value = '';
      sendButton.disabled = true;

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: history })
        });

        const data = await response.json();
        if (!response.ok) {
          addMessage(data.error || 'Server error', 'bot');
        } else {
          const content = data.reply;
          history.push({ role: 'assistant', content });
          addMessage(content, 'bot');
        }
      } catch (err) {
        addMessage('Unable to reach server.', 'bot');
      }

      sendButton.disabled = false;
      promptInput.focus();
    }

    sendButton.addEventListener('click', sendMessage);
    promptInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
      }
    });
  </script>
</body>
</html>
'''

def build_prompt(messages):
    prompt_parts = [SYSTEM_PROMPT, '']
    for item in messages:
        role = item.get('role')
        content = item.get('content', '').strip()
        if not content:
            continue
        if role == 'system':
            prompt_parts.append(f'System: {content}')
        elif role == 'user':
            prompt_parts.append(f'User: {content}')
        elif role == 'assistant':
            prompt_parts.append(f'Nova: {content}')
    prompt_parts.append('Nova:')
    return '\n'.join(prompt_parts)

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True)
    if not data or 'messages' not in data:
        return jsonify({'error': 'Invalid request'}), 400

    messages = data['messages']
    prompt = build_prompt(messages)

    try:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(DEVICE)
        generated = model.generate(
          **inputs,
          max_new_tokens=256,
          do_sample=True,
          temperature=0.7,
          top_p=0.9,
          pad_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return jsonify({'reply': reply})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
