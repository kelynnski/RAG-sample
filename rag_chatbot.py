import openai
from datetime import date
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

class chatbot:
   def __init__(self, personality, cite_sources = False, keep_history = False, source_distance = 1, temperature=1):
       self.personality = personality
       self.cite_sources = cite_sources
       self.keep_history = keep_history
       self.source_distance = source_distance
       self.temperature = temperature
       personality = "Call Center"

       instruction = "Pretend you are a human being who works at the ASU Online Call Center. "
       instruction += "Your goal is to answer my questions about Arizona State University. "
       instruction += "Please be friendly, helpful, and informative. "
       instruction += "Do not say anything about being an AI or a language model in your responses. "
       instruction += "Do not say anything about getting information from the available sources, as that sounds like a robot and you are a human."
       instruction += "\nToday is " + date.today().strftime("%B %d, %Y") + "."
       instruction += " The current semester is Spring 2024."

       response = "Please enter your question."
       self.messages = [
           {"role": "user", "content": instruction},
           {"role": "assistant", "content": response},
       ]

   @property
   def cite_sources(self):
       return self._cite_sources

   @cite_sources.setter
   def cite_sources(self, cite_sources):
       if(cite_sources is True):
           print("Warning! Sources within the response may be incorrect.")
       self._cite_sources = cite_sources

   def create_context(
       self, question, df, max_len=1800, size="ada"
   ):
       sources = []
       #Get the embeddings for the question
       q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
       #Get the distances from the embeddings
       df['distances'] = distances_from_embeddings(q_embeddings, df['embedding'].values, distance_metric='cosine')
       returns = []
       cur_len = 0

       #Sort by distance and add the text to the context until the context is too long
       for i, row in df.sort_values('distances', ascending=True).iterrows():
           #Add the length of the text to the current length
           cur_len += row['n_tokens'] + 4
           #If the context is too long, break
           if cur_len > max_len:
               break
           # If the distance is too far, break
           if row['distances'] > self.source_distance:
               break
           #Else add it to the text that is being returned
           text = row['content']

           if self.cite_sources:
               text += "\nSource: " + row['url']
           returns.append(text)
           source = (row['url'], row['distances'])
           sources.append(source)

       #Return the context
       return ("\n\n###\n\n".join(returns), sources)

   def answer_question(
           self,
           question,
           model="gpt-3.5-turbo",
           max_len=1800,
           size="ada",
           debug=False,
           max_tokens=150,
           stop_sequence=None
   ):
       #Answer question based on most similar context from dataframe texts
       (context, sources) = self.create_context(
           question,
           df,
           max_len=max_len,
           size=size,
       )

       #sources is a list of tuples, where the first element is the url and the second is the distance
       #we want to remove url duplicates and keep the closest distance
       #we also want to sort by distance
       sourceUrls = []
       sourceDistances = []
       for source in sources:
           if source[0] not in sourceUrls:
               sourceUrls.append(source[0])
               sourceDistances.append(source[1])
       sourcesSet = list(zip(sourceUrls, sourceDistances))

       #If debug, print the raw model response
       if debug:
           print("Context:\n" + context)
           print("\n\n")
       try:
           prompt = "Answer the question based on the provided content."
           prompt += "If the context is not relevant to the question, do not reference it."
           prompt += f" If the question can't be answered based on the context, start your answer with \"I'm not entirely sure. \" Please answer in complete sentences. \n\n Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"

           if self.cite_sources:
               prompt += "\nIn your answer, please state the source (url) provided in the section of the context you used to answer the question. Sections are separated by \"###\" and the source is listed at the end of the section. If you did not use a source, do not include a source."
           prompt += ". Your response should not be longer than a few sentences."

           tempMessages = self.messages.copy()
           tempMessages.append({"role": "user", "content": prompt})
           response = openai.ChatCompletion.create(
               model=model,
               messages=tempMessages,
               temperature=self.temperature
           )
           if self.keep_history:
               self.messages.append({"role": "user", "content": question})
               self.messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
           return (response["choices"][0]["message"]["content"], sourcesSet)

       except Exception as e:
           print(e)
           return ("", "")

   def generate_questions(self, model="gpt-3.5-turbo", max_tokens=150):
       try:
           #Retrieve the most recent user question from the conversation history
           question = next((message['content'] for message in reversed(self.messages) if message['role'] == 'user'), None)
           if not question:
               raise ValueError("No user question found in the conversation history.")
           #Construct the prompt to generate related questions
           prompt = f"Generate three related questions based on the following user query: '{question}'"

           #Make the API call to generate related questions
           response = openai.ChatCompletion.create(
               model=model,
               messages=[{"role": "system", "content": "Provide three related questions to this question: " + question},  # System message to set the behavior of the assistant
                         {"role": "user", "content": question}],  #User's original question
               max_tokens=max_tokens,
               temperature=0.7
           )

           if self.keep_history:
               self.messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
           return response["choices"][0]["message"]["content"], self.messages

       except Exception as e:
           print(e)
           return "", []
