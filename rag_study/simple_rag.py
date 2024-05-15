import os

from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
import re


#原始文档存储路径
origin_data_path = 'data/'
chroma_vec_save_path = 'data/vector_db/chroma'


## 文档清洗加载
def text_clean(text):
    '''
    清洗文本
    :param text:
    :return:
    '''
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    text = re.sub(pattern, lambda match: match.group(0).replace('\n', ''),text)

    #删•和空格
    text = text.replace("*","").replace(" ","")
    return text

def load_data_from_file_path(file_path):
    '''
    从指定路径里面加载文档信息
    :param file_path:
    :return:
    '''
    texts = []

    for root,dirs,files in os.walk(file_path):
        for file in files:
            file_path = os.path.join(root,file)
            file_type = file_path.split('.')[-1]
            if file_type == 'pdf':
                loader = PyMuPDFLoader(file_path)
            elif file_type == 'md':
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise NotImplementedError("不支持当前文件类型：%s" % file_type)
            texts.extend(loader.load())
    texts = [text_clean(t) for t in texts]
    return texts

def split_text(text,chunk_size=500,ovelap_size=50):
    '''
    分割文本
    RecursiveCharacterTextSplitter(): 按字符串分割文本，递归地尝试按不同的分隔符进行分割文本。
    CharacterTextSplitter(): 按字符来分割文本。
    MarkdownHeaderTextSplitter(): 基于指定的标题来分割markdown 文件。
    TokenTextSplitter(): 按token来分割文本。
    SentenceTransformersTokenTextSplitter(): 按token来分割文本
    Language(): 用于 CPP、Python、Ruby、Markdown 等。
    NLTKTextSplitter(): 使用 NLTK（自然语言工具包）按句子分割文本。
    SpacyTextSplitter(): 使用 Spacy按句子的切割文本。
    :param text:
    :param chunk_size: 分块的最大长度
    :param ovelap_size: 文档重叠长度
    :return:
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=ovelap_size
    )
    split_docs = text_splitter.split_text(text)
    return split_docs

def get_llm_result(text,model="zhipu",temperature=0.1):



if __name__ == '__main__':
    file_name = r'./data/pumpkin_book.pdf'
    load_pdf(file_name)