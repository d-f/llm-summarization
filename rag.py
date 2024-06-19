import datasets
import langchain_core.runnables
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import langchain_core
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WikipediaLoader
from langchain_community.document_loaders import TextLoader
import argparse
from pathlib import Path
from typing import Type, List


def parse_cla() -> Type[argparse.ArgumentParser]:
    """
    parses command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-custom_ds", action="store_true")
    parser.add_argument("-md_dir", type=Path)
    parser.add_argument("-wiki_query", type=str)
    parser.add_argument("-max_docs", type=str)
    parser.add_argument("-temp", type=int)
    parser.add_argument("-rep_pen", type=float)
    parser.add_argument("-max_new_tok", type=int)
    parser.add_argument("-num_ex", type=int)
    parser.add_argument("-chunk_size", type=int)
    parser.add_argument("-chunk_overlap", type=int)
    parser.add_argument("-llm_path", type=str)
    parser.add_argument("-em_model_name", type=str)
    parser.add_argument("-tok_path", type=str)
    parser.add_argument("-load_4bit", action="store_true")
    parser.add_argument("-quant_type", type=str)
    parser.add_argument("-dtype", type=str)
    parser.add_argument("-dbl_quant", action="store_true")
    return parser.parse_args()


def wiki_loader(query:str, max_docs:List) -> Type[WikipediaLoader]:
    """
    loads wikipedia data

    keyword arguments:
    query -- query searched in the wikipedia data
    max_docs -- maximum number of documents retrieved
    """
    loader = WikipediaLoader(query=query, lang="en", load_max_docs=max_docs)
    return loader    


def custom_loader(md_dir:Path) -> List:
    """
    creates TextLoader from folder with .md files
    """
    doc_list = []
    for path in md_dir.iterdir():
        loader = TextLoader(file_path=str(path))
        docs = loader.load()
        doc_list += docs
    return doc_list


def prepare_docs(chunk_size:int, chunk_overlap:int, docs:List) -> List:
    """
    splits characters in chunks of length chunk_size

    keyword_arguments 
    chunk_size -- amount of characters in each chunk
    chunk_overlap -- amount of characters overlapping between adjacent chunks
    docs -- list of documents to chunk
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = splitter.split_documents(docs)
    return chunked_docs


def chain(text_generation_pipeline:Type[pipeline]) -> Type[langchain_core.runnables.RunnableSequence]:
    """
    create chain of prompt, model and StrOutputParser
    """
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    prompt_template = """
    ### Instruction: Summarize the following text labeled as text to summarize based on your knowledge. 

    Use the following context to help:
    {context}

    Text to summarize:
    {input_txt}
    
    ### SUMMARY:
    """
    prompt = PromptTemplate(
        input_variables=["context", "input_txt"],
        template=prompt_template,
    )
    llm_chain = prompt | llm | StrOutputParser()
    return llm_chain


def load_model(
        load_in_4bit:bool, 
        bnb_4bit_quant_type:str, 
        bnb_4bit_compute_dtype:str, 
        bnb_4bit_use_double_quant:bool, 
        llm_path:str
        ) -> Type[AutoModelForCausalLM]:
    """
    loads Causal LLM

    keyword arguments:
    load_in_4bit -- 4-bit precision
    bnb_4bit_quant_type -- quantization data type {nf4, fp4}
    bnb_4bit_compute_dtype -- data type for computation
    bnb_4bit_use_double_quant -- nested quantization
    llm_path -- path to folder with llm file in hf format
    """
    quant_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            )
    model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            quantization_config=quant_config
        )
    return model


def rag_inf(rag_chain:Type[langchain_core.runnables.RunnableSequence], input_text:str) -> str:
    """
    predicts with RAG model
    """
    return rag_chain.invoke(input_text["content"])


def rag_examples(
        num_examples:int, 
        rag_chain:Type[langchain_core.runnables.RunnableSequence], 
        ds:Type[datasets.arrow_dataset.Dataset]
        ):
    """
    predicts a certain amount of examples with the RAG model
    """
    break_int = 0
    for ex in ds:
        if break_int == num_examples:
            break
        print(rag_inf(rag_chain=rag_chain, input_text=ex))
        break_int += 1


def main():
    args = parse_cla()
    if args.custom_ds:
        docs = custom_loader(md_dir=args.md_dir)
    else:
        loader = wiki_loader(query=args.wiki_query, max_docs=args.max_docs)
        docs = loader.load()

    chunked_docs = prepare_docs(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, docs=docs)
    db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name=args.em_model_name))

    model = load_model(
        load_in_4bit=args.load_4bit, bnb_4bit_quant_type=args.quant_type, bnb_4bit_compute_dtype=args.dtype,
        bnb_4bit_use_double_quant=args.dbl_quant, llm_path=args.llm_path
        )
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=args.temp,
        do_sample=True,
        repetition_penalty=args.rep_pen,
        return_full_text=True,
        max_new_tokens=args.max_new_tok,
    )

    llm_chain = chain(text_generation_pipeline=text_generation_pipeline)
    retriever = db.as_retriever()
    rag_chain = {"context": retriever, "input_txt": RunnablePassthrough()} | llm_chain

    ds = datasets.load_dataset("webis/tldr-17")

    rag_examples(num_examples=args.num_ex, rag_chain=rag_chain, ds=ds["train"])


if __name__ == "__main__":
    main()
