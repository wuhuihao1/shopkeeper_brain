import re, os, json
from typing import Tuple, List, Dict, Any
from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.utils.markdown_util import MarkdownTableLinearizer

class DocumentSplitNode(BaseNode):
    name = 'document_split_node'
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        文档切分的核心逻辑入口
        Args:
            state: 节点状态

        Returns:
            更新状态
        """
        config = self.config
        # 1. 参数校验
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state, config)
        # 2. 切分（一级策略：根据md文档中的标题来切分）多个章节（章节：标题之间的内容）
        sections: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)
        # 3. 二次切分或者合并 大于max_length就切,小于且同源就合并
        final_section = self._split_and_merge(sections, max_content_length, min_content_length)
        # 4. 组装成chunk
        final_chunks = self._assemble_chunks(final_section)
        # 5. 备份
        self._back_up(final_chunks, state)
        # 6. 更新状态
        state['chunks'] = final_chunks
        return state

    def _back_up(self, final_chunks: List[Dict[str, Any]], state: ImportGraphState) -> None:
        """
        对chunks进行备份,存放到json文件
        Args:
            final_chunks:最后组装好的chunk
            state:节点状态

        Returns:

        """
        #获取存放目录
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            #exist_ok属性,开启后允许路径已经存在文件夹
            os.makedirs(local_dir, exist_ok=True)
            #拼装路径
            output_path = os.path.join(local_dir, "chunks.json")
            # 写入文件
            with open(output_path, "w", encoding="utf-8") as f:
                #允许输出中文ensure_ascii
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")





    def _assemble_chunks(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        组装chunks函数
        拼装方法:title + \n\n + body
        Args:
            sections: 切分,合并好的章节列表
        Returns:
            拼装好的chunks列表
        """
        final_chunks = []
        for section in sections:
            body = section['body']
            title = section['title']
            parent_title = section['parent_title']
            file_title = section['file_title']


            content = f"{title}\n\n{body}"

            final_chunks.append({
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title,
                "content": content,
            })
        self.logger.info(f'最终切割后能够进入到嵌入节点的chunk个数:{len(final_chunks)}')
        return final_chunks




    def _split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int) -> \
            List[Dict[str, Any
            ]]:
        """
        切分大章节,合并小章节
        Args:
            sections: 所有经过一级切分后的章节
            max_content_length: 超过该值就切分
            min_content_length: 最小长度,小于该值就合并

        Returns:
            先切后合
        """
        # 1.切分
        current_section = []
        for section in sections:
            current_section.extend(self._split_long_section(section, max_content_length))

        # 2. 合并
        final_sections = self._merger_short_section(current_section, min_content_length)
        return final_sections


    def _merger_short_section(self, current_sections: List[Dict[str, Any]], min_content_length: int) -> List[Dict[str, Any]]:
        """
        合并短章节内容
        短章节来源:
        1.一级标题切分后本来就短
        2.经过_split_long_section函数切分后导致变短
        合并策略:
        1.必须同源:父标题一致
        2.合并方长度要小于min_content_length

        Args:
            sections: 经过切分后的章节列表
            min_content_length: 最小长度

        Returns:
            合并后的章节列表

            使用贪心累加算法,当前章节长度不够,就加上下一个同源章节
        """
        #默认取当前第一个章节
        if current_sections:
            current_section = current_sections[0]
        else:
            current_section = {}
        final_sections = []
        #遍历合并
        for next_section in current_sections[1:]:
            #判断章节父标题是否和下一个章节父标题一致(判断同源
            same_parent = (current_section['parent_title'] == next_section['parent_title'])
            #如果同源且长度过小就合并
            if same_parent and len(current_section['body']) < min_content_length:
                #合并,把当前标题改为父标题
                current_section['body'] = (current_section['body'].rstrip()+ '\n\n' + next_section['body'].lstrip())
                current_section['title'] = current_section['parent_title']
            else:
                final_sections.append(current_section)
                #更新指针
                current_section = next_section
        final_sections.append(current_section)
        return final_sections


    def _split_long_section(self, section: Dict[str, Any], max_content_length: int) -> List[Dict[str, Any]]:
        """
        切分单个长章节函数
        Args:
            section: 当前章节
            max_content_length: 最大长度
        Returns:
            返回切分后的章节列表
        """
        # 1. 获取section对象的属性
        body = section['body']
        title = section['title']
        parent_title = section['parent_title']
        file_title = section['file_title']

        # 防止title过长,截取前80个字符
        if len(title) > 80:
            title = title[:80]

        if '<table>' in body:
            self.logger.info('检查到section中有表格')
            # MarkdownTableLinearizer使用降维转译法转为一维的自然语言
            body = MarkdownTableLinearizer.process(body)

        # 2. 获取标题前缀
        title_pre = f'{title}\n\n'
        # 3. 获取总长度【标题(前缀)+body】
        total = len(body) + len(title_pre)
        # 4. 判断总长度是否超过阈值
        if total <= max_content_length:
            return [section]
        # 5. 能切分的内容长度计算出来（body）
        body_len = max_content_length - len(title_pre)
        #标题太长无法切分
        if body_len <= 0:
            return [section]
        # 6. 切分(6.1 用谁切:LangChain的切分器:递归切分器 6. 去切谁:body)
        # 6.1 切分器对象  # 1. chunk_size:chunk块的大小  2.chunk_overlap块与块之间的重叠的字符数  3. separators：分割符 【"\n\n",'\n',' ',''】
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=body_len,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "？", "！", "；", ".", "?", "!", ';', " ", ""],
            keep_separator=True, #切分后保留分隔符
        )
        # 6.2 切分器对象切分
        sections = text_spliter.split_text(body)
        # 6.3 判断,如果切分后值为1,说明未切分,内部无分隔符直接返回
        if len(sections) == 1:
            return [section]
        # 6.4 遍历
        sub_sections = []
        for index, body_new in enumerate(sections):
            #组装新的section
            sub_sections.append({
                "body": body_new,
                "title": f'{title}_{index + 1}',
                "parent_title": parent_title,
                "file_title": file_title,
            })

        # 7. 返回
        return sub_sections



    def _validate_state(self, state: ImportGraphState, config) -> Tuple[str, str, int, int]:
        """
        校验文档的参数,返回必要的值
        Args:
            state:
            config:

        Returns:

        """
        self.log_step("step1", "切分文档的参数校验以及获取...")
        #1.获取md_content
        md_content = state.get('md_content')
        # 2. 统一换行符\r该\n
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        # 3. 获取文件标题
        file_title = state.get('file_title')
        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 \
                or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")
        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:
        """
        parent_title:封装的原因主要为了后面短section在合并的时候有一个判断标准（同源：同一个父标题）
        根据标题来切分（# {1,6}都有可能）
        Args:
            md_content: 文档内容
            file_title: 文档标题

        Returns:
            List[Dict]:切分后的多个章节
        """
        in_fence = False #是否有代码块
        sections = []
        body_lines = []
        current_title = ''
        hierarchy = [''] * 7 #标题层级追踪数组
        current_level = 0

        def _flush():
            """
            打包selection
            {
                body: 收集到的所有行,
                title: 文档标题,
                parent_title: 父标题
                file_title: 文档标题
            }
            Returns:
                如果current_title没有,body有,打包为selection
                如果current_title有,body没有,打包为selection
                如果current_title有,body有,打包为selection
                如果current_title没有,body没有,不打包为selection

            """
            # 1. 处理内容行
            body = '\n'.join(body_lines)
            #current_title或者body俩者任一有值就执行
            if current_title or body:
                #初始化parent_title
                parent_title = ''
                #从当前等级前一个遍历,找父title
                for i in range(current_level -1, 0, -1):
                    #必须为非空字符
                    if hierarchy[i]:
                        #找到就赋值并弹出
                        parent_title = hierarchy[i]
                        break
                #如果parent_title为空
                if not parent_title:
                    #如果有当前current_title就给current_title作为父标题,否则说明全文无标题,给文档标题
                    parent_title = current_title if current_title else file_title
                #组装字典并放入sectionsList
                sections.append({
                    "body": body,
                    "title": current_title if current_title else file_title,
                    "parent_title": parent_title,
                    "file_title": file_title
                })

        # 1. 根据\n切分md_content
        md_lines = md_content.split('\n')
        # 2. 定义正则（正则的规则是从MD中找标题#{1,6}）():捕获组:产生三个group(0) group(1):#(1)#(6) group(2)标题的内容
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
        # 3. 遍历切分后md_lines
        for line in md_lines:
            # 3.1 检测代码块边界（``` 或 ~~~）代码块要留下来
            if line.strip().startswith('```') or line.strip().startswith('~~~'):
                #代码块有俩个```,所以用这个别写死
                in_fence = not in_fence
            # 3.2 判读是否要走正则,走了说明匹配的是标题否则是正文
            match = heading_re.match(line) if not in_fence else None
            # 3.3 判断math 是否有值,有值说明匹配到了不为空的标题
            if match:
                #将body_lines收集到的行封装为selection
                _flush()
                current_title = line #当前标题就是这个匹配到的字段
                level = len(match.group(1)) #group1匹配的是#,根据#个数计算当前等级
                #保存等级和标题信息
                current_level = level
                hierarchy[level] = current_title

                for i in range(level + 1, 7):
                    #清空当前标题之后的内容,防止影响后续内容读写
                    hierarchy[i] = ''
                body_lines = []
            else:
                body_lines.append(line)
        _flush()
        return sections

if __name__ == '__main__':
    document_split_node = DocumentSplitNode()

    md_path = r"W:\test\PythonProject\shopkeeper_brain\knowledge\processor\import_processor\万用表RS-12的使用_origin\hybrid_auto\万用表RS-12的使用_origin.md"

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_content": md_content,
        "file_title": "万用表的使用",
        "file_dir": r"W:\test\PythonProject\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }
    document_split_node.process(init_state)


