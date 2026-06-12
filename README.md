# multi_rag
### This module facilitates testing RAG pipelines on Local machines with chroma_db, with text embeddings from 'bge-base-en-v1.5' and image embeddings from 'clip-vit-b-32'

### Currently this module supports pdf, docx, xlsx, png, jpg, jpeg and txt file formats

#### User can give the path of the file to the embed function, which sets up the chroma_db/ folder for the embeddings while temp/ folder gets set up to mimic the actual database to store the data chunks.

#### Retrive function takes query as input and gives out a dictionary of 'text','tables','images'

#### query function takes query as input and returns the answer and retrieved data as output

#### One needs to have gemini api key to query, but embedding and retrieval part is completely local