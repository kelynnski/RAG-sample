import openai
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

class chatbot:
   def __init__(self, cite_sources = False, keep_history = False, source_distance = .3, temperature=1):
       self.cite_sources = cite_sources
       self.keep_history = keep_history
       self.source_distance = source_distance
       self.temperature = temperature

       instruction = "Your goal is to assess information about web tools, and use the relevant information to answer specific questions about the tool. "
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

   """
   def generate_hypothetical_document(self, question, model="gpt-3.5-turbo-instruct", temperature=1):
       system_prompt = "Even if you don't know the answer respond to this question with output from a typical AI tool website: "
       system_prompt += question
       print(f"Generating document with prompt: {system_prompt}")

       try:
           response = openai.Completion.create(
               engine=model,
               prompt=system_prompt,
               temperature=temperature,
               max_tokens=250,
               top_p=1.0,
               frequency_penalty=0.0,
               presence_penalty=0.0
           )

           return response.choices[0].text.strip()
       except Exception as e:
           print(e)
           return ""
   """

   def create_context(self, question, df, max_len=1800, size="ada"):
       """
       #Generate hypothetical document
       hypothetical_document = self.generate_hypothetical_document(question)
       print()
       print(f'Question: {question}')
       print(f'This is the hypothetical document: {hypothetical_document}')
       print()

       #combine hypothetical document with question for embeddings
       combined_input = hypothetical_document + ". " + question
       """
       sources = []

       #Get the embeddings for the combined input
       q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
       #Get the distances from the embeddings
       df['distances'] = distances_from_embeddings(q_embeddings, df['Embeddings'].values, distance_metric='cosine')

       returns = []
       cur_len = 0

       #Sort by distance and add the text to the context until the context is too long
       for i, row in df.sort_values('distances', ascending=True).iterrows():
           #Add the length of the text to the current length
           cur_len += row['n_tokens'] + 4
           #If the context is too long, break
           if cur_len > max_len:
               break
           #If the distance is too far, break
           if row['distances'] > self.source_distance:
               break
           #Else add it to the text that is being returned
           text = row['AI Cleaned Text']
           if self.cite_sources:
               text += "\nSource: " + row['URL']
           returns.append(text)
           source = (row['URL'], row['distances'])
           sources.append(source)
 
       #Return the context
       return ("\n\n###\n\n".join(returns), sources)

   def answer_question(
           self,
           question,
           df,
           tool_name,
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
           print('This is the context and question:')
           print(question)
           print()
           print(context)
           print()

           prompt = "Do your best to answer the question. I am providing some additional context to help answer the question, but if you are already "
           prompt += "familiar with the tool, feel free to provide information you know. "
           prompt += f"The name of the tool we need information about is {tool_name}."
           prompt += "If you are unsure about what the answer is, state that you are not entirely sure, and give a hypothetical or probable answer. "
           prompt += f"Additional context: {context} \n\n Question you need to answer: {question}"

           if self.cite_sources:
               prompt += "\nIn your answer, please state the source (url) provided in the section of the context you used to answer the question. Sections are separated by \"###\" and the source is listed at the end of the section. If you did not use a source, do not include a source."

           response = openai.ChatCompletion.create(
               model=model,
               messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
               temperature=self.temperature
           )
           if self.keep_history:
               self.messages.append({"role": "user", "content": question})
               self.messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})

           return (response["choices"][0]["message"]["content"], sourcesSet)

       except Exception as e:
           print(e)
           return ("", "")
