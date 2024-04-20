import openai
import tiktoken

class chatbot:
   def __init__(self, keep_history = False, temperature=1):
       self.keep_history = keep_history
       self.temperature = temperature

       instruction = "You provide the most relevant topics from provided text, because the provided text was scraped from websites and is messy. "
       instruction += "The kind of content you want to include is information in complete sentences that conveys a clear message. Avoid including text "
       instruction += "in your response that is likely to have been scraped from a website header and footer, such as addresses, phone numbers, and common website "
       instruction += "footer elements like 'Join the team', 'Book a demo', 'Contact sales', etc. Try to look out for words that might have been buttons on a website "
       instruction += "that don't make sense to include as part of the main ideas."

       response = "I provide the most relevent information from text that I am provided. I focus on the main ideas of the text and avoid information "
       response += "that was likely scraped from website headers, footers, navigation menus, and website buttons."

       prompt = "Extract the relevant information from this text:"
       prompt += "\n### Text content ###\n"
       prompt += "html About | Insync Solutions  \u008f Email AI Chat AI for Customer Support Chat AI for Ecommmerce Agent AI InSync Browser Co-Pilot (beta) Why Insync  \u008f Technology Reporting & Insights Integrations Resources  \u008f About Blog Case Studies Contact sales Book a demo Contact sales Book a demo About We provide a cutting edge support automation solution with conversational AI for online lending and retail eCommerce companies. Join the team  Join the team  Join the team  About us Insyncai was founded by Raj Ramaswamy and Ashish Parnami with the mission to make Conversational AI more accessible and easy to implement for enterprises and without requiring any time or effort commitment from their side. Our powerful Conversational AI solution helps our clients drive sales, better customer acquisitions and scale their support operations at a fraction of the cost. Join the team  Meet the team Raj Ramaswamy CEO & Co-Founder  \u0099 Ashish Parnami CTO & Co-Founder  \u0099 Scott Sapire VP Sales  \u0099 Girish Nair Lead Architect  \u0099 Manish Jain Manager Data Science  \u0099 Anisha Jayan Tech Lead  \u0099 Prasanna Kumar K Editorial Project Manager  \u0099 Shrisha S Bhat Sn. Technical Product Manager  \u0099 Investors & Advisors Watertower Ventures  \u0099 ¡\u0086 BAM Ventures  \u0099 ¡\u0086 Foothill Ventures  \u0099 ¡\u0086 Arka Venture Labs  \u0099 ¡\u0086 Tuoc Luong  \u0099 ¡\u0086 Vivek Sharma  \u0099 ¡\u0086 Vishal Makhijani  \u0099 ¡\u0086 Jesse Bridgewater  \u0099 ¡\u0086 Contact us USA office Riverpark Tower, 333 W San Carlos St, San Jose, CA 95110 +1 888-291-0379 info@insyncai.com India office Indiqube Orion 4th Floor, 24th Main, HSR Layout, Bangalore-560102 India Interested in joining the team? Learn more  \u0092 USA office Riverpark Tower, 333 W San Carlos St, San Jose, CA 95110 +1 888-291-0379 info@insyncai.com India office Indiqube Orion 4th Floor, 24th Main, HSR Layout, Bangalore-560102 India Solutions Email AI Chat AI for Customer Support Chat AI for Ecommerce Agent AI Why Insync Technology Reporting & Insights Integrations Product About Blog Case Studies Privacy Policy Copyright 2024 @ InSync All Rights Reserved  Security Overview"
       prompt += "\n### End text content ###\n"

       example_response = "We provide a cutting edge support automation solution with conversational AI for online lending and retail eCommerce companies. "
       example_response += "Insyncai was founded by Raj Ramaswamy and Ashish Parnami with the mission to make Conversational AI more accessible and easy to implement for enterprises and without requiring any time or effort commitment from their side. Our powerful Conversational AI solution helps our clients drive sales, better customer acquisitions and scale their support operations at a fraction of the cost."

       second_prompt = "Extract the relevant information from this text:"
       second_prompt += "\n### Text content ###\n"
       second_prompt += "html Co-Pilot Solutions   Email AI Chat AI for Customer Support Chat AI for Ecommmerce Agent AI InSync Browser Co-Pilot (beta) Why Insync   Technology Reporting & Insights Integrations Resources   About Blog Case Studies Contact sales Book a demo Contact sales Book a demo Your Trusty Browser Co-Pilot Your AI-powered browser sidekick to navigate through complex web pages, sort through products, get product information, summarize articles, pdfs and more. The Co-Pilot uses advanced AI to understand the contents and uses generative AI to answer your questions in seconds. Install for Chrome  Get information on products in seconds The InSync Co-Pilot goes through entire web pages including embedded PDFs, so you can get all the information you need before you buy. Install for Chrome  Summarize articles, reviews snd more InSyncs Co-Pilot saves you time by summarizing long content articles, research, and more, so you can get the gist in seconds. Install for Chrome  Designed for shoppers, students, programmers & more.. InSyncs Co-Pilot runs on all kinds of web pages from E-commerce to Education to Q&A websites like Stack Overflow and more. Install for Chrome  Research @ the speed of thought InSyncs Co-Pilot helps you speed up your research by going through research documents, PDFs, etc and helping you get to the key information faster. Install for Chrome  InSync cares deeply about user safety and security and takes great care when handling your data. InSync does not retain or store any PII and all data is anonymized. Please do not share any personal information, including passwords, credit card or banking details. This plugin uses Artificial Intelligence (AI) and there may be inaccuracies or unintended bias in any results provided. By continuing, you agree to Open AI's and InSync's Terms & Privacy Policy . For any questions or feedback on Co-Pilot please contact us at copilot@insyncai.com . Get Always On - White Glove Service Get powerful business insights generated by our AI and enhanced by our Data Science team delivered to you on a weekly basis from your Insync AI Expert.   Book a demo USA office Riverpark Tower, 333 W San Carlos St, San Jose, CA 95110 +1 888-291-0379 info@insyncai.com India office Indiqube Orion 4th Floor, 24th Main, HSR Layout, Bangalore-560102 India Solutions Email AI Chat AI for Customer Support Chat AI for Ecommerce Agent AI Why Insync Technology Reporting & Insights Integrations Product About Blog Case Studies Privacy Policy Copyright 2024 @ InSync All Rights Reserved  Security Overview"
       second_prompt += "\n### End text content ###\n"

       second_example_response = "Your AI-powered browser sidekick to navigate through complex web pages, sort through products, get product information, summarize articles, pdfs and more. The Co-Pilot uses advanced AI to understand the contents and uses generative AI to answer your questions in seconds. "
       second_example_response += "Get information on products in seconds The InSync Co-Pilot goes through entire web pages including embedded PDFs, so you can get all the information you need before you buy. Summarize articles, reviews snd more InSyncs Co-Pilot saves you time by summarizing long content articles, research, and more, so you can get the gist in seconds. "
       second_example_response += "Install for Chrome  Designed for shoppers, students, programmers & more.. InSyncs Co-Pilot runs on all kinds of web pages from E-commerce to Education to Q&A websites like Stack Overflow and more. Research @ the speed of thought InSyncs Co-Pilot helps you speed up your research by going through research documents, PDFs, etc and helping you get to the key information faster. InSync cares deeply about user safety and security and takes great care when handling your data. InSync does not retain or store any PII and all data is anonymized. Please do not share any personal information, including passwords, credit card or banking details. This plugin uses Artificial Intelligence (AI) and there may be inaccuracies or unintended bias in any results provided."

       self.messages = [
           {"role": "user", "content": instruction},
           {"role": "assistant", "content": response},
           {"role": "user", "content": prompt},
           {"role": "assistant", "content": example_response},
           {"role": "user", "content": second_prompt},
           {"role": "assistant", "content": second_example_response},
       ]

   def answer_question(self, question, model="gpt-3.5-turbo"#, max_tokens=150
                       ):
       try:
           tokenizer = tiktoken.get_encoding("cl100k_base")
           #Add the user's question to the messages
           self.messages.append({"role": "user", "content": question})
           input_tokens = sum(len(tokenizer.encode(message["content"])) for message in self.messages)

           response = openai.ChatCompletion.create(
               model=model,
               messages=self.messages,
               temperature=self.temperature,
               #max_tokens=max_tokens
           )

           #Extract the response content
           response_content = response.choices[0].message.content
           output_tokens = len(tokenizer.encode(response_content))

           if self.keep_history:
               #Save the assistant's response to the history
               self.messages.append({"role": "assistant", "content": response_content})
           return response_content, input_tokens, output_tokens

       except Exception as e:
           print(f"Error during chat completion: {e}")
           #Return empty response and zero tokens in case of an exception
           return "", 0, 0
