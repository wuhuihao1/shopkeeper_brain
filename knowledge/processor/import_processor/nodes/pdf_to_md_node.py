# 1.内置包
import subprocess, time, json
from typing import Tuple
from pathlib import Path
# 2.三方包
# 3.自己的包
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, PdfConversionError

#定义节点类
class PdfToMdNode(BaseNode):
    name='pdf_to_md_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """节点处理逻辑入口"""
        # 接受文件的path,调用mineru的分析工具处理成md文件
        # 1.获取文件的路径以及输出目录
        import_file_path_obj, ouput_dir_path_obj = self._validate_state(state)

        # 2.输入两个路径,执行mineru解析（命令： mineru -p input_path -o output_dir --source=local）
        processed_code = self._execute_mineru_parse(import_file_path_obj, ouput_dir_path_obj)
        if processed_code != 0:
            raise PdfConversionError(node_name=self.name, message='MinerU解析失败')
        #获取md路径
        md_path = self.get_md_path(import_file_path_obj, ouput_dir_path_obj)
        state['md_path'] = md_path

        return state
    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """导入节点图谱状态,输出目录路径"""
        self.log_step('stpe1', '准备校验和获取解析文件的路径')
        # 1.获取解析文件路径
        import_file_path = state.get('import_file_path', '')
        # 2. 判断是否为空
        if not import_file_path:
            #报错,节点参数错误
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str, message='解析文件的路径不能为空')
        # 3. 标准化解析文件的路径
        import_file_path_obj = Path(import_file_path)
        # 如果路径参数不存在
        if not import_file_path_obj.exists():
            #继续报错
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str, message='解析文件的路径不存在')
        # 5. 获取输出文件目录
        ouput_dir_path = state.get('file_dir', '')
        if not ouput_dir_path:
            #设置默认输出文件夹位置是解析文件的父类
            ouput_dir_path = import_file_path_obj.parent
        #标准化
        ouput_dir_path_obj = Path(ouput_dir_path)
        if not ouput_dir_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str,
                                  message='输出文件夹的路径不存在')
        self.logger.info(f"解析的文件路径{import_file_path}")
        self.logger.info(f"输出文件夹目录{ouput_dir_path}")
        #返回校验通过的Path对象
        return import_file_path_obj, ouput_dir_path_obj

    def _execute_mineru_parse(self, import_file_path_obj, ouput_dir_path_obj) -> int:
        """
        执行mineru的命令函数,传入输入输出路径,返回code码0表示成功,1表示失败
        :param import_file_path_obj:
        :param ouput_dir_path_obj:
        :return:
        """
        # 定义cmd
        cmd = [
            'mineru',
            '-p',
            str(import_file_path_obj),
            '-o',
            str(ouput_dir_path_obj),
            '--source',
            'local'
        ]
        #定义开始时间,用于计算分析时间
        start_time = time.time()
        #定义一个子进程来执行cmd命令
        proc = subprocess.Popen(cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,  #输出的是文字
            encoding='utf-8', #定义字符集
            errors='replace', #特殊字符用?菱形表示
            bufsize=1 #实时输出。按行输出遇到\n换行符 就将日志产生出来
        )
        #实时打印日志
        for line in proc.stdout:
            print(f'mineru解析时产生的日志:{line}')
        #主线程等待子进程,得到一个状态码
        processed_result = proc.wait()
        end_time = time.time()
        if processed_result == 0:
            print(f'pdf解析成功,耗时{end_time - start_time:.2f}s')
        else:
            print(f'pdf解析失败')
        return processed_result
    def get_md_path(self, import_file_path_obj: Path, ouput_dir_path_obj: Path) -> str:
        """
        解析获取md文件路径
        md_path= D:\develop\develop\workspace\temp_dir\万用表的使用\hybrid_auto\万用表的使用.md
        Path里的三个属性: name: 名字+后缀; stem: 名字+无后缀; suffix: 纯后缀
        :param import_file_path_obj:
        :param ouput_dir_path_obj:
        :return:
        """
        file_name = import_file_path_obj.stem
        md_path = str(ouput_dir_path_obj / file_name / "hybrid_auto" / f'{file_name}.md')
        return md_path
if __name__ == '__main__':
    setup_logging()
    #构建节点实例
    pdf_to_md_node = PdfToMdNode()
    #初始化实例对象
    init_state: ImportGraphState = {
        "import_file_path": r"W:\test\PythonProject\shopkeeper_brain\docs\万用表RS-12的使用.pdf",
        "file_dir": r"W:\test\PythonProject\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }
    result = pdf_to_md_node.process(init_state)
    #indent4个空格缩进,ensure_ascii允许输出中文
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(result_str)