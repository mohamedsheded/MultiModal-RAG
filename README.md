# Multi-modal RAG

## Table of Contents
1. [Introduction](#multi-modal-rag)
2. [Approaches](#approaches)
   - [Option 1](#option-1)
   - [Option 2](#option-2)
   - [Option 3 (Our Choice)](#option-3-our-choice)
3. [Architecture Design](#architecture-design)
   - [Data Extraction Code Example](#data-extraction-code-example)
4. [Tools Used](#tools-used)
5. [Demo and Output](#demo-and-output)
   - [Checking Retrieval](#checking-retrieval)
   - [Checking RAG Pipeline](#checking-rag-pipeline)
6. [User Interface](#user-interface)
7. [Model Enhancement](#model-enhancement)
8. [References](#references)
   
## Multi-modal RAG
Many documents contain a mixture of content types, including text and images. However, information captured in images is often lost in most RAG applications.

With the emergence of multimodal LLMs, we can now leverage images in RAG effectively. Below are three possible approaches:

## Approaches
### Option 1:
- Use multimodal embeddings (such as CLIP) to embed images and text.
- Retrieve both using similarity search.
- Pass raw images and text chunks to a multimodal LLM for answer synthesis.

### Option 2:
- Use a multimodal LLM to produce text summaries from images.
- Embed and retrieve Summaries.
- Pass text chunks to an LLM for answer synthesis.

### Option 3 (Our Choice):
- Use a multimodal LLM to produce text summaries from images.
- Embed and retrieve image summaries with a reference to the raw image.
- Pass raw images and text chunks to a multimodal LLM for answer synthesis.

We have chosen **Option 3** as it allows for efficient image-text integration while maintaining high retrieval accuracy.

## Architecture Design
1. Use the **Unstructured Library** to extract data from PDFs, including text, images, and tables.
2. Use a multimodal LLM to produce text summaries from images.
3. Embed and retrieve image summaries with a reference to the raw image.
4. Pass raw images, tables, and text chunks to a multimodal LLM for answer synthesis.

![image](https://github.com/user-attachments/assets/67f88c56-d3f7-41b3-84cd-244b9ae62b2d)

### Data Extraction Code Example:
```python
raw_pdf_elements = partition_pdf(
    filename="/content/attention is all you need.pdf",  # mandatory
    strategy="hi_res",  # mandatory to use ``hi_res`` strategy
    extract_images_in_pdf=True,  # mandatory to set as ``True``
    extract_image_block_types=["Image", "Table"],  # optional
    extract_image_block_to_payload=False,  # optional
    extract_image_block_output_dir="extracted_data",  # optional - only works when ``extract_image_block_to_payload=False``
)
```

After partitioning the PDF, we obtained:
- **83 text chunks** as narrative text.
- **4 tables**.
- **7 images**.

## Tools Used
- **Unstructured Library**: For data ingestion and extraction.
- **Groq Cloud**: For LLM/vLLM processing.
- **Chroma DB**: As a vector database.
- **Gradio**: For building an interactive UI tool.
- **Models** : For summarizing tables/Text `gemma2-9b-it` for summarizing Images `llama-3.2-90b-vision-preview` or `llama-3.2-11b-vision-preview`
## Demo and Output
### Checking Retrieval
We verified the system's retrieval performance using the following five queries:

1. **How is the scaled dot product attention calculated?**
2. **What is the BLEU score of the model in English to German translation (EN-DE)?**
3. **How long were the base and big models trained?**
4. **Which optimizer was used when training the models?**
5. **Show me a picture that shows the difference between Scaled Dot-Product Attention and Multi-Head Attention.**

All queries were retrieved correctly after reviewing the original paper.
**Note:** The **Groq Cloud API** does not support multi-image retrieval for generation. While we successfully retrieved multiple images, the open-source vLLM used only allows one image per request in the preview release. Requests with multiple images will return a **400 error**. 
![image](https://github.com/user-attachments/assets/919e432c-17ac-45c7-bdc2-f3a0fa3768a6)

**Alternative Approach:** To support multi-image retrieval for generation, alternative models such as **OpenAI (GPT-4V)** and **Gemini** can be used, as they support processing multiple images in a single request.
Our evaluations confirm that the pipeline effectively retrieves and processes multimodal content, providing accurate and meaningful results as found on the original paper after eyeballing the retrieved docs to searching with the paper.

![image](https://github.com/user-attachments/assets/c14e145d-522e-4913-a910-8a77ac662192)

![image](https://github.com/user-attachments/assets/9f3ca41c-159a-4a64-b603-732fafe3641f)
### Checking Generation
![image](https://github.com/user-attachments/assets/e436511c-ab2c-4f50-8270-904280f0f261)

[Demo on provided](https://drive.google.com/file/d/1Yy8-hUZhNN-RxgNnrboQtUB3JhhFH_UT/view?usp=sharing)

[Demo-more examples](https://drive.google.com/file/d/1cMSDoiCxOldLwJOpnRhAW14MVL4HIszD/view?usp=sharing)

### Checking RAG Pipeline
To validate the RAG pipeline, we ensured that:
- Data extraction from PDFs (text, images, and tables) was successful.
- The multimodal LLM correctly generated summaries for images.
- Image summaries were embedded and retrieved with high accuracy.
- The system successfully synthesized answers by combining text and image data.

## User Interface
To improve user confidence in the retrieval system, we implemented an option to display the retrieved context with the final answer. This feature allows users to verify the source of information and ensure accuracy.
![image](https://github.com/user-attachments/assets/7f1c6287-6223-450e-bbf8-9f6b796dda5a)


The UI, built using **Gradio**, provides:
- A toggle option to show or hide the retrieved context.
- A structured view of text, tables, and images retrieved.

This enhancement ensures transparency in retrieval and improves trust in the model's responses.

## Model Enhancement
To address existing challenges and improve retrieval accuracy, we propose integrating **Agentic RAG**, which leverages autonomous agent-based methods for better query understanding and document synthesis. **Agentic RAG** can:

- Dynamically refine queries based on context for more precise retrieval.
- Automatically cross-check retrieved images and text to eliminate irrelevant data.
- Enhance answer synthesis by combining insights from multiple sources (web search Tools) as in **CRAG**.
- Improve multi-step reasoning and contextual awareness in multimodal retrieval tasks **ReAct Method**.

By integrating **Agentic RAG / CorrectiveRAG**, we can overcome retrieval limitations and ensure that both textual and visual content contribute effectively to the final response generation.
![image](https://github.com/user-attachments/assets/0c4567e0-d66b-4c26-80ae-70a9239a7ad7)

## References
[LangChain Blogpost](https://blog.langchain.dev/semi-structured-multi-modal-rag/)

For more about agents: [My GitHub repo for implementing agentic patterns](https://github.com/mohamedsheded/Agentic-design-patterns)

For Agentic RAG notebook: [My GitHub repo](https://github.com/mohamedsheded/LangGraph-projects/tree/main/Agentic%20Rag)
