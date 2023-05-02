import tiktoken

model = "gpt-3.5-turbo-0301"
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


f = open("system_role", "r")
system_role=f.read()

import json
path='/graphics/scratch2/staff/Hassan/genius_crawl/genius_data/Ac dc.json'
with open(path) as f:
   data = json.load(f)

song=data['songs']


totall_tokens_in=0
totall_tokens_out=0
print('number of songs for this artist: ', len(data['songs']))

for song in data['songs']:

    lyric=song['lyrics']

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": 'Prioritise rule 2 and don\'t use generic terms. Do you understand?'},
        {"role": "assistant", "content": 'Yes, I understand. Let\'s get started!'},
        {"role": "user", "content": lyric},
    ]

    #print(f"{num_tokens_from_messages(messages, model)} prompt tokens counted.")
    totall_tokens_in += num_tokens_from_messages(messages, model)

#approximating the output of chatgpt
for song in data['songs']:

    lyric=song['lyrics']

    messages = [
        {"role": "user", "content": lyric},
    ]

    #print(f"{num_tokens_from_messages(messages, model)} prompt tokens counted.")
    totall_tokens_out += num_tokens_from_messages(messages, model)

print('total input token', totall_tokens_in)
print('total output token', totall_tokens_out)

this=(totall_tokens_in/len(data['songs'])) * 0.000002 + ((totall_tokens_out/len(data['songs'])) * 0.000002)
print('avg cost for a single song:  {} dollars: '.format(this))


print('for 250k songs ~ ', this * 250000)
