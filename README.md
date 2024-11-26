# Expert Chat Agent – RAG Based Application
### Project description

This project was implemented to showcase the implementation of a RAG based application.  The purpose is simple. Create an application that would allow a human to interact with a **CHAT AGENT** that becomes an expert on a specific topic that has been “fed” to the application via the *Admin* site.  

The idea is to allow a user to ask very specific questions to a chat agent, that would otherwise, imply the user to sort and look specific information in documents that can potentially be hundreds of pages long. 

The primarily objective to build this application is to be used for a group of compensation experts to asks questions about a company’s proxy statements.
Application architecture

The architecture of this application is relatively simple and involves a two-step process.

-	*Admin site:* Admin uploads a PDF document, that the embeddings model will turn into vectors for future use

-	*User site:* User simply asks a question, the application conducts a similarity search leveraging on the vector database, which then produces an augmented prompt (with the additional context) for the main LLM to use to generate the response.

The following diagram shows the general architecture diagram, followed by a brief description about the services used and their role in the application:

<img width="100%" align="center" alt="image" src="https://github.com/user-attachments/assets/cb2534d6-ee84-408f-b4f2-64cee2bcd2df">
