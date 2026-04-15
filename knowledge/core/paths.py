import os

# __file__ 当前文件路径
# os.path.dirname 当前文件所在父目录core
# .. 回到上一级目录knowledge 也就是根目录
KNOWLEDGE_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LOCAL_BASE_DIR=os.path.join(KNOWLEDGE_ROOT, "temp_data")

FRONT_PAGE_DIR=os.path.join(KNOWLEDGE_ROOT, "front")

def get_local_base_dir():
    return LOCAL_BASE_DIR

def get_front_page_dir():
    return FRONT_PAGE_DIR
