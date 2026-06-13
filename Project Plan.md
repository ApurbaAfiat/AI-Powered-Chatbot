

## **My Task: Build a simple AI-powered chatbot that can answer questions about a financial policy document** 

## **1. Extract Data** 

- Read the financial policy document ( **provided with the email** ). 

- Pull out important information (like budget, debt, or infrastructure details) 

- Make sure I keep track of where I found the info (page number or section is enough). 

## **2. Set Up Database for Search** 

- Store the main points or sections in a database (like faiss, chormadb, pinecone) for quick searching. 

- When the user asks a question, use vector search to find the closest matching section or answer. 

## **3. Build the Chatbot** 

## **My chatbot should:** 

- Understand simple user questions about the document. 

- Use **conversation memory** so it remembers what the user just asked (so if someone asks “What about debt?” right after another question, the bot knows the topic). 

- Give answers using the information found through the knowledge base search. 

- Respond clearly and helpfully. 


