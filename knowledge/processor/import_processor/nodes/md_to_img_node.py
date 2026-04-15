import re, time, base64
from collections import deque #双端队列
from dataclasses import dataclass #dataclass装饰器
from logging import Logger #日志
from typing import List, Dict, Tuple, Set, Optional, Deque #类型
from pathlib import Path #路径对象

from openai import OpenAI #OpenAI对象

from knowledge.processor.import_processor.base import BaseNode, setup_logging, T #基础节点类,开始日志
from knowledge.processor.import_processor.state import ImportGraphState #状态属性
from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError #定义错误类型
from knowledge.utils.client.ai_clients import AIClients #openAi客户端
from knowledge.utils.client.storage_clients import StorageClients

@dataclass #装饰器,后续使用该类时不需要重写__init__方法 __repr__方法
class ImageContext:
    """
    定义图片上下文类型
    """
    head: str
    pre_text: str
    post_text: str

@dataclass
class ImageInfo:
    """
    定义图片的完整信息
    图片的名字: 作为图片的存储容器key保存
    图片的地址: 提供给vlm使用,提供给minio使用
    图片的上下文信息: 提供给vlm使用
    """
    name: str #名字
    path: str #路径
    image_context: ImageContext #图片上下文信息

class _MdFileHandler:
    """
    文件处理类
    处理md_path,md_path,图片目录
    备份新的md_content
    """
    def __init__(self,logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def validate_and_read_md(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
        校验md路径是否合法,读取图片所在目录
        Args:
            state: 上一个节点更新后的状态
        Returns:
            Tuple[str, Path, Path]
        """
        # 1. 从state中获取md_path
        md_path = state.get('md_path', '')
        # 2. 非空判断
        if not md_path:
            raise StateFieldError(node_name=self.node_name, field_name='md_path', expected_type=str, message='md_path不能为空!')
        # 3. Path标准化
        md_path_obj = Path(md_path)
        # 4. 判断路径是否存在
        if not md_path_obj.exists():
            raise StateFieldError(node_name=self.node_name, field_name='md_path',message='md文件不存在!')
        # 5. 读取md_content
        with open(md_path_obj, 'r', encoding='utf-8') as f:
            md_content = f.read()
        # 6. 获取图片目录
        image_dir = md_path_obj.parent / 'images'
        # 7. 返回
        return md_content, md_path_obj, image_dir

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        """
        备份文件
        Args:
            md_path_obj:
            new_md_content:

        Returns:

        """
        # with_name用于替换文件名字,父目录保持不变,返回一个新路径
        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise FileProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


class _ImageScanner:
    """
    图片扫描类
    1.根据图片目录,得到该图片下有效的图片文件
    2.去到md文件中定位图片的位置
    3.获取该图片在md中的上下文内容
    4.最终组装所有图片的上下文内容
    """
    def __init__(self,logger: Logger):
        self.logger = logger
    def scan_imgs_dir(self, img_dir_obj: Path, md_content: str, image_extensions: Set[str], img_content_length: int) -> List[ImageInfo]:
        """
        核心方法,扫描images文件夹
        1.扫描指定图片目录
        2.遍历每一个图片文件获取到图片的上下文
        3.将每一个图片的上下文封装到一个容器(imageInfo)中
        4.将imageInfo组成list返回
        Args:
            img_dir_obj:图片文件夹所在位置
            md_content: md的内容
            image_extensions: 上传图片允许的后缀
            img_content_length: 上传文件允许的最大长度

        Returns: ImageInfo组成的列表

        """
        img_info_list = []
        # 1.遍历图片目录
        #iterdir方法,path对象内置,用于遍历目录里所有内容
        for img_file in img_dir_obj.iterdir():
            # 1.1 过滤掉子目录
            if not img_file.is_file():
                continue
            # 1.2  过滤掉不合法的图片文件
            if img_file.suffix not in image_extensions:
                continue
            # 1.3 找该图片的上下文
            md_context = self._find_context(img_file.name, md_content, img_content_length)
            if not md_context:
                self.logger.error(f'未找到{img_file.name}图片的上下文')
                continue
            # 1.4 封装ImageInfo对象并且放到容器中
            img_info_list.append(ImageInfo(name=img_file.name, path=str(img_file), image_context=md_context))
        # 2. 最终返回
        return img_info_list
    def _find_context(self, md_name: str, md_content: str, img_content_length: int) -> Optional[ImageContext]:
        """
        查找图片上下文
        Args:
            md_name: 图片名
            md_content: md文件内容
            img_content_length: md文件最大允许长度
        Returns: 返回图片链接的上下文 或者 None (没找到图片)
        """
        #1.通过正则从md文件中找到这张图
        # . 任意字符 * 0次或者多次  \[ \] \( \) ?非贪婪模式  escape（a.png）
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(md_name) + r".*?\)")

        # 2. 按行切割md_content
        md_lines = md_content.split("\n")
        # 3. 遍历每一行以及对应的行索引
        for idx, line in enumerate(md_lines): #idx当前行下标
            # 3.1 当前行不是当前图片
            if not pattern.search(line):
                continue
            # 3.2 当前行包含当前图片
            # 上文
            # 上文标题的索引作为起始索引(取不到)
            head, pre_index = self._find_heading_up(md_lines, idx) #head标题, pre_index上文的下标
            pre_lines = md_lines[pre_index+1: idx]
            #上下文长度限制函数_extract_limited_context
            pre_text = self._extract_limited_context(pre_lines, img_content_length, direction="front")
            # 下文
            # 下文标题的索引作为结束索引
            post_index = self._find_heading_down(md_lines, idx)  # head标题, pre_index上文的下标
            post_lines = md_lines[idx+1: post_index]
            #上下文长度限制函数_extract_limited_context
            post_text = self._extract_limited_context(post_lines, img_content_length, direction="back")
            return ImageContext(head=head, pre_text=pre_text, post_text=post_text)
        return None

    def _find_heading_up(self, md_lines: List[str], idx: int) -> Tuple[str, int]:
        """
        找图片上文方法
        Args:
            md_lines: 整个文档列表
            idx: 当前图片所在序号
        Returns: 当前图片最近的上文标题内容+索引
        """
        #遍历文档通过下标反向寻找图片上文,直到第一行
        for i in range(idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]): #通过#号匹配标题
                return md_lines[i], i
        #没找到标题返回空字符
        return "", -1

    def _find_heading_down(self, md_lines: List[str], idx: int) -> int:
        """
        找图片下文方法
        Args:
            md_lines: 整个文档列表
            idx: 当前图片所在序号
        Returns: 下文最近标题的索引
        """
        #遍历文档通过下标反向寻找图片上文,直到第一行
        for i in range(idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]): #通过#号匹配标题
                return i
        return len(md_lines)


    def _extract_limited_context(self, lines: List[str], img_content_length: int, direction: str) -> str:
        """
        限制上下文长度函数
        截取段落,如果段落字符长度超过最大字符则保留当前段落,不再截取其他段落
        段落的规则：
        ①：自然而然的段落 获取切分后的内容 如果是""空字符串
        ②：人为设计其他图片作为段落（其它图片不要）
        Args:
            lines: 上下文list
            img_content_length: 最大字符数
            direction: 方向,front上文,back下文
        Returns: str:上（下）文的内容
        """
        current_paragraph = []
        paragraphs = []
        selected = []
        total = 0
        # 1. 遍历截取的行
        for line in lines:
            # 1.1 定义自然而然段落的规则
            is_blank_line = not line.strip()
            # 1.2 定义人为设计的图片段落规则匹配![]()
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )
            # 1.3 当前行是空行或者其它图片行
            if is_other_image or is_blank_line:
                #先保存之前的内容
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue
            # 1.4  当前行不是空行也不是其它图片行
            current_paragraph.append(line)
        # 2. 处理最后的行
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))
        # 处理front, 反转(就近原则), 因为要优先保留距离图片最近的段落
        if direction == "front":
            paragraphs.reverse()
        # 3. 遍历段落列表(判断长度，已经最终选择留下哪些段落)
        for paragraph in paragraphs:
            if total + len(paragraph) > img_content_length and selected:
                break
            selected.append(paragraph)
        # 处理front 反转（保证收集到的顺序和原文档中顺序一致，方便VLM参考）
        if direction == "front":
            selected.reverse()
        # 4. 将最终段落列表中的段落转成一个字符串
        return "\n\n".join(selected)

class _VLMSummarizer:
    """
    vlm调用类
    通过调用模型生成图片摘要信息
    """
    def __init__(self, logger: Logger, requests_per_minute: int):
        self.logger = logger
        self.requests_per_minute = requests_per_minute

    def _summary_all(self, document_name: str, img_info_list: List[ImageInfo], vl_model: str) -> Dict[str, str]:
        """
        为图片生成摘要
        Args:
            document_name: 文档名称
            img_info_list: imgInfo组成的list
            vl_model: 模型名称

        Returns: 字典{图片名称, 图片摘要}

        """
        summaries = {}
        request_timestamps: Deque[float] = deque()
        # 1. 获取VLM客户端
        try:
            client = AIClients.get_vlm_client()
        except Exception as e:
            for img_info in img_info_list:
                self.logger.error('模型无法调用!')
                summaries[img_info.name] = '暂无摘要'
            return summaries
        # 2.调用VLM 为每一张图片生成摘要
        for img_info in img_info_list:
            self._enforce_rate_limit(request_timestamps, self.requests_per_minute)
            summaries[img_info.name] = self._summary_one(document_name, img_info, client, vl_model)
        self.logger.info(f"生成{len(summaries)}条图片摘要")
        return summaries

    def _summary_one(self, document_name: str, img_info: ImageInfo, vlm_client: OpenAI, vl_model: str) -> str:
        """
        调用VLM模型为当前图片生成摘要信息
        Args:
            document_name: 文档名称
            img_info: 图片信息
            vlm_client: vlm客户端
            vl_model: vlm模型

        Returns: 图片摘要

        """
        # 1. 构造VLM需要的上下文（标题名、上文内容、下文内容）
        parts = [p for p in
                (img_info.image_context.head, img_info.image_context.pre_text, img_info.image_context.post_text) if p]

        # 2. 构建最终的上下文
        final_context = '\n'.join(parts) if parts else '暂无上下文'
        # 3. 根据图片地址获取到图片的内容（二进制字节流）---文本协议认识（base64编码）--->解码（‘utf-8’）--->字符串（文本协议能传输） ---- 根据收到字符串解码（二进制字节流 还原图片内容）
        try:
            with open(img_info.path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
        except IOError as e:
            self.logger.error(f'读取图片{img_info.name}失败,失败原因:{e}')
            return '暂无图片描述'
        # 4. 利用vlm客户端调用VLM模型
        try:
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"图片摘要生成失败 {img_info.path}: {e}")
            return "暂无图片描述"

    def _enforce_rate_limit(self,timestamps: Deque[float], max_requests: int,window: int = 60):
        """
        限制函数,限制最多能同时处理多少张图片, 多了就sleep
        Args:
            timestamps:
            max_requests: ,
            window:
        Returns: None

        """
        now = time.time()
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()
        if len(timestamps) >= max_requests:
            sleep_dur = window - (now - timestamps[0])
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()

        timestamps.append(now)

class _ImageUploader:
    """
        图片上传类：
        1. 将本地图片上传到MinIO，得到该图片在MinIO中可访问的远程地址
        2. 替换md中的摘要和图片地址
    """
    def __init__(self, logger: Logger):
        self.logger = logger

    def _upload_all(self, object_dir_name: str, img_info_list: List[ImageInfo], minio_url: str,
                    minio_bucket_name: str) -> Dict[str, str]:
        remote_urls = {}
        # 1. 得到MinIO客户端
        try:
            client = StorageClients.get_minio_client()
        except Exception as e:
            #没客户端就直接把本地路径返回
            for img_info in img_info_list:
                remote_urls[img_info.name] = img_info.path
            return remote_urls
        # 2. 遍历上传每一个
        for img_info in img_info_list:
            #包含桶名字和图片名
            object_name = f"{object_dir_name}/{img_info.name}"
            # 2.1 上传图片到MinIO
            client.fput_object(
                minio_bucket_name, object_name, img_info.path)
            # 2.2 自己拼装路径
            # http://192.168.200.140:9000/桶名/对象名
            self.logger.info(f'成功将图片{img_info.name}上传到MinIO中')
            remote_urls[img_info.name] = f'{minio_url}/{minio_bucket_name}/{object_name}'
        return remote_urls

    def _update_md(self, md_content, summaries, remote_urls):
        """
        更新md文件,通过正则匹配图片url然后更换
        Args:
            md_content: md文件
            summaries: 字典[图片名, 摘要]
            remote_urls: 字典[图片名, 桶地址]

        Returns:
            新md
        """
        # ()是正则的捕获组,group(0)是所有内容(1)(2)...分别代表()内的东西出现次数
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer (match: re.Match) -> str:
            """
            匹配函数,返回替换内容
            Args:
                match:
            Returns:
                ![摘要](远程图片地址)
            """
            for img_name, img_summary in summaries.items():
                #取到当前匹配项在文档中的路径和名字
                origin_img_path = match.group(2)
                img_name_in_md = Path(origin_img_path).name
                if img_name_in_md == img_name:
                    #组装返回值
                    return f"![{img_summary}]({remote_urls[img_name]})"
            #未匹配到返回原样
            return match.group(0)
        return pattern.sub(replacer , md_content)



    def upload_and_replace(self,  object_dir_name: str, md_content: str, img_info_list: List[ImageInfo],
                           summaries: Dict[str, str],
                           minio_url: str, minio_bucket_name: str):
        """
        核心函数
        上传图片到miniIO
        更新md中的图片地址以及摘要
        Args:
            object_dir_name: minio对象目录
            md_content: md文档内容
            img_info_list: 图片信息List
            summaries: 摘要字典
            minio_url: minio的地址
            minio_bucket_name: 桶名
        Returns:
        """
        # 1. 上传
        remote_urls = self._upload_all(object_dir_name, img_info_list, minio_url, minio_bucket_name)
        # 2. 更新
        md_content = self._update_md(md_content, summaries, remote_urls)
        return md_content
class MarkDownToImgNode(BaseNode):
    """
        图片处理节点,主程序入口：
        1. 得到四个类的实例对象
        2. 分别调用四个实例对象的处理方法
    """
    def __init__(self):
        super().__init__()  # 显示调用父类的构造方法
        self._md_file_handler = _MdFileHandler(self.logger, self.name)
        self._img_scaner = _ImageScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger, self.config.requests_per_minute)
        self._img_uploader = _ImageUploader(self.logger)
    name = 'md_to_img_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点入口逻辑
        Args:
            state: 状态

        Returns: 更新状态

        """
        config = self.config
        # 1. 操作md_file_handler
        self.log_step("step1", "读取MD内容、路径以及图片的目录")
        md_content, md_path_obj, image_dir = self._md_file_handler.validate_and_read_md(state)
        # 1.1 判断图片目录不存在
        if not image_dir.exists():
            state['content'] = md_content
            return state
        # 2. 操作_img_scaner
        self.log_step("step2", "准备开始扫描图片目录")
        image_info_list = self._img_scaner.scan_imgs_dir(image_dir,md_content,config.image_extensions, config.img_content_length)

        # 3. 操作_vlm_summarizer
        self.log_step("step3", "利用VLM提取摘要")
        summaries: Dict[str, str] = self._vlm_summarizer._summary_all(md_path_obj.stem, image_info_list, config.vl_model)
        # 4. 操作_img_uploader
        self.log_step("step4", "上传文件到MinIO,且更新MD")
        new_md_content = self._img_uploader.upload_and_replace(md_path_obj.stem, md_content, image_info_list,
                                                               summaries,
                                                               config.get_minio_base_url(),
                                                               config.minio_bucket)
        state['md_content'] = new_md_content

        # 5. 备份
        self.log_step("step5", "备份md文件")
        self._md_file_handler.backup(md_path_obj, new_md_content)

        return state
if __name__ == '__main__':
    setup_logging()
    md_img_node = MarkDownToImgNode()
    init_state = {
        "md_path": r"W:\test\PythonProject\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表RS-12的使用\hybrid_auto\万用表RS-12的使用.md"
    }
    md_img_node.process(init_state)


