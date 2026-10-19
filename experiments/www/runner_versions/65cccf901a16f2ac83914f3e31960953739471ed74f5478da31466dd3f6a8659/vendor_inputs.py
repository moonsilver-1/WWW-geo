"""Content-preserving formatting for the pinned vendor chat templates."""
from copy import deepcopy


def tokenizer_options(key):
    # Restore Pixtral's native Tekken splitting in the converted checkpoint.
    return {'fix_mistral_regex': True} if key == 'pixtral12b' else {}


def prepare_messages(key, messages):
    result = deepcopy(messages)
    if key == 'pixtral12b' and result[0]['role'] == 'system':
        instruction = result.pop(0)['content']
        result[0]['content'].insert(0, {'type': 'text', 'text': instruction + '\n\n'})
    if key == 'smolvlm2_22b':
        for message in result:
            if isinstance(message['content'], str):
                message['content'] = [{'type': 'text', 'text': message['content']}]
    return result


def reference_text(key, statement):
    return (' ' if key == 'smolvlm2_22b' else '') + statement


def phi_prompt(messages, add_generation_prompt):
    result = ''
    for message in messages:
        content = message['content']
        if isinstance(content, list):
            content = ''.join('<image>' if part['type'] == 'image' else part['text'] for part in content)
        if message['role'] == 'assistant':
            content = '<nothink>' + content
        result += f"<|im_start|>{message['role']}<|im_sep|>{content}<|im_end|>"
    if add_generation_prompt:
        result += '<|im_start|>assistant<|im_sep|><nothink>'
    return result
