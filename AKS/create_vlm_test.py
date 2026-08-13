# -*- coding: utf-8 -*-
import os
import logging
import base64
import hashlib
import hmac
import json
from typing import Union, Tuple, Optional

import requests
import time
import uuid
from threading import Lock
from pathlib import Path
from string import Template


MEP_TIMEOUT = int(os.getenv("MEP_TIMEOUT", 240))
FIRST_TOKEN_TIMEOUT = int(os.getenv("FIRST_TOKEN_TIMEOUT", 120000))

import json
import re

def parse_llm_json(response: str):
    """
    解析LLM返回的JSON，兼容：
    - ```json{...}```
    - ```json\n{...}\n```
    - ```{...}```
    - 前后解释文本
    """

    response = response.strip()

    # 去掉 markdown fence
    response = re.sub(r"^```(?:json)?", "", response.strip(), flags=re.IGNORECASE)
    response = re.sub(r"```$", "", response.strip())

    # 找到最外层JSON
    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Cannot find JSON in response:\n{response}")

    json_str = match.group(0)

    data = json.loads(json_str)

    return data

def load_text(txt_path: str) -> str:
    """
    读取 txt 文件内容。

    Args:
        txt_path: txt 文件路径

    Returns:
        文件全部内容（字符串）
    """
    return Path(txt_path).read_text(encoding="utf-8").strip()

class MepClient:
    """ 请求Mep服务的通用客户端

        Attributes:
            __url : str
                MEP的ELB地址
            appID : str
                请求的appID
            secret_key : str
                appID设置的秘钥
            b_id: str
                应用标识
            flow_id: str
                业务标识

    """

    init_lock = Lock()

    def __init__(self, mep_elb_url, appid, secret_key, b_id, flow_id, stream=False,
                 first_token_timeout_ms=FIRST_TOKEN_TIMEOUT):
        logging.info(f"stream={stream}")
        if stream:
            mep_elb_url = mep_elb_url.replace("/service", "/predict")
        self.__url = mep_elb_url
        self.appID = appid
        self.secret_key = secret_key
        self.b_id = b_id
        self.flow_id = flow_id
        self.stream = stream
        self.first_token_timeout_ms = first_token_timeout_ms

    def execute(self, request_data: dict, method="POST", callback=None) -> Union[dict, str, None]:
        request_id = str(uuid.uuid4())[:8]
        headers = self.__make_headers(method, request_id)
        payload = self.__make_request_body(request_data, request_id)
        canonicalQueryString = f"?bId={self.b_id}&flowId={self.flow_id}&uuId={request_id}" if self.stream else ""
        url = self.__url + canonicalQueryString
        try:
            if self.stream:
                session = requests.Session()
                r = session.post(url, headers=headers, json=payload, stream=self.stream, verify=False)
                results = []
                start_time = time.time()
                first_token_received = False

                detection_window = ""
                max_window_size = 2000
                min_cycle_length = 50
                max_cycle_length = 500

                repeat_threshold = 15
                current_repeat_count = 0
                last_content = None

                for line in r.iter_lines():
                    if line:
                        if not first_token_received:
                            current_time = time.time()
                            elapsed_ms = (current_time - start_time) * 1000
                            if elapsed_ms > self.first_token_timeout_ms:
                                logging.warning(f"首token超时，超时时间: {self.first_token_timeout_ms}ms")
                                return None

                        line_text = line.decode('utf-8').strip()

                        try:
                            event_data = json.loads(line_text)
                            event_type = event_data.get("event")

                            if event_type == "data" or event_type == "think":
                                content = event_data.get("content", "")

                                if not content.strip():
                                    continue

                                first_token_received = True

                                if content == last_content:
                                    current_repeat_count += 1
                                    if current_repeat_count >= repeat_threshold:
                                        logging.warning(f"检测到连续重复内容，停止流式请求: {content[:50]}...")
                                        return None
                                else:
                                    current_repeat_count = 0
                                    last_content = content

                                detection_window += content

                                if len(detection_window) > max_window_size:
                                    detection_window = detection_window[-max_window_size:]

                                if len(detection_window) >= min_cycle_length * 3:
                                    cycle_detected, cycle_content = self._detect_long_cycle(detection_window,
                                                                                            min_cycle_length,
                                                                                            max_cycle_length)
                                    if cycle_detected:
                                        logging.warning(f"检测到长循环重复模式，停止流式请求。循环内容: {cycle_content}")
                                        return None

                                results.append(content)

                                if callback and callable(callback):
                                    callback(content)

                            elif event_type == "finish":
                                break

                        except json.JSONDecodeError:
                            logging.warning(f"Failed to parse line as JSON: {line_text}")
                            continue

                return "".join(results)
            else:
                r = requests.request(
                    method="POST",
                    url=url,
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=MEP_TIMEOUT,
                    verify=False,
                    stream=self.stream
                )
                try:
                    response = json.loads(r.text)
                    print(11111111, response)
                except json.JSONDecodeError as e:
                    logging.info(f"parse json result error: {e}. stream={self.stream}")
                    return r.text
                return self.__parse_response(response)
        except Exception as e:
            logging.info(f"get result from mep error {e}. stream={self.stream}")
            return "" if self.stream else {}

    def _detect_long_cycle(self, text: str, min_cycle_length: int, max_cycle_length: int) -> Tuple[bool, Optional[str]]:
        """
        检测文本中是否存在长循环模式

        Args:
            text (str): 要检测的文本
            min_cycle_length (int): 最小循环长度
            max_cycle_length (int): 最大循环长度

        Returns:
            Tuple[bool, Optional[str]]: (是否检测到循环, 循环内容)
        """
        text_len = len(text)

        # 从较短的循环开始检测，这样能更快发现问题
        for cycle_len in range(min_cycle_length, min(max_cycle_length + 1, text_len // 3)):
            # 检查最后的文本是否包含至少3次重复的循环
            if text_len >= cycle_len * 3:
                # 取最后的部分进行检测
                recent_text = text[-cycle_len * 3:]

                # 检查是否存在重复模式
                pattern = recent_text[:cycle_len]

                # 验证这个模式是否在后续文本中重复出现
                repeat_count = 0
                for i in range(0, len(recent_text) - cycle_len + 1, cycle_len):
                    if recent_text[i:i + cycle_len] == pattern:
                        repeat_count += 1
                    else:
                        break

                # 如果发现至少3次重复，认为是循环
                if repeat_count >= 3:
                    logging.warning(f"检测到长度为{cycle_len}字符的循环，重复{repeat_count}次")
                    return True, pattern

        return False, None

    def __make_headers(self, method, request_id) -> dict:
        """构造请求的请求头

            Args:
                method (str) : 请求体的方法

            Returns:
                headers (Dict) : 请求需要携带的header

        """

        httpMethod = method
        urlPath = "/predict" if self.stream else "/service"
        canonicalQueryString = f"bId={self.b_id}&flowId={self.flow_id}&uuId={request_id}" if self.stream else ""
        httpPayload = ""
        timestamp = str(round(time.time() * 1000))

        stringToSign = httpMethod + "&" + urlPath + "&" \
                       + canonicalQueryString + "&" + httpPayload \
                       + "&appid=" + self.appID + "&timestamp=" + timestamp

        stringToSign = stringToSign.encode('utf-8')
        secretKey = self.secret_key.encode('utf-8')
        signature = base64.b64encode(hmac.new(secretKey, stringToSign,
                                              digestmod=hashlib.sha256).digest()).decode("utf-8")

        accessToken = ("CLOUDSOA-HMAC-SHA256 appid=" + self.appID + ", timestamp=" + timestamp +
                       ", signmode=easy, signature=\"" + signature + "\"")
        headers = {'Content-Type': "application/json", 'Authorization': accessToken}
        return headers

    def __make_request_body(self, request_data: dict, request_id) -> dict:
        """构造请求mep elb的请求体

            Args:
                request_data (Dict) : 请求的服务接受的请求体

            Returns:
                payload: 包含路由信息的请求体

        """

        payload = {}
        payload['version'] = "1.0" if request_data.get('version') is None else request_data.get('version')
        if "version" in request_data:
            del request_data["version"]
        payload['data'] = {}
        payload['meta'] = {}
        payload['data'] = request_data
        if self.stream:
            return payload
        payload['meta']['bId'] = self.b_id
        payload['meta']['flowId'] = self.flow_id
        payload['meta']['isPressureTest'] = False
        payload['meta']['uuId'] = request_id
        return payload

    @staticmethod
    def __parse_response(response: dict) -> dict:
        """提取mep响应中包含的服务响应体

        """

        result = response['result']
        if result == {}:
            logging.info(f"ori response:{response}")
        return result

def image_to_base64(image_path):
    """
    将本地图片转换为纯 Base64 编码字符串（不含 data:image 头部），
    以便在外部通过 f"data:image/jpeg;base64,{base64_str}" 进行精准拼接。

    :param image_path: 本地图片文件的路径 (str)
    :return: 纯 Base64 编码字符串
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片文件: {image_path}")

    # 读取文件并进行 base64 编码
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        base64_data = base64.b64encode(image_bytes)
        base64_str = base64_data.decode("utf-8")

    return base64_str

def vlm_check(vlm_client, un_tar_path, img):
    if not os.path.exists(os.path.join(un_tar_path, img)):
        raise FileNotFoundError(f"图像文件 {img} 不存在于路径 {un_tar_path} 中")

    # vlm_prompt = VL_OBSERVATION_PROMPT_JD.format(
    #     tt=resp_json['title'],
    #     dctt=resp_json['dctt'],
    #     entity=resp_json['result'][-1]
    # )
    with open("./prompt/summary.json", "r", encoding="utf-8") as f:
        summary_prompt = json.load(f)

    # full_text = "抓取网页全文"
    full_text = load_text("context/xiaomi.txt")[:5000]
    # print("full_text", full_text)
    # 构建总结的提示词
    summary_sysprompt = summary_prompt["system"]
    summary_userprompt = Template(
        summary_prompt["user"]
    ).substitute(
        full_text=full_text
    )

    # print("summary_userprompt", summary_userprompt)
    base64_str = image_to_base64(os.path.join(un_tar_path, img))

    vlm_data = {
        "param": {
            "temperature": 0.1,
            "max_tokens": 8192,
            "frequency_penalty": 0.2,
            "top_p": 0.95,
        },
        "messages": [
            {
                "role": "system",
                "content": summary_sysprompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{summary_userprompt}"
                    },
                ]
            }
        ]
    }
    print("=" * 100)
    print(json.dumps(vlm_data, ensure_ascii=False, indent=2))
    print("=" * 100)

    response = vlm_client.execute(vlm_data)
    print("response", response)
    print("response_type", type(response))
    print(len(full_text))
    # resp_json = json.loads(response[7:].rstrip('```'))\
    # 格式有报错的可能
    resp_json = parse_llm_json(response)

    global_summary = resp_json["global_summary"]
    key_entities = resp_json["key_entities"]
    print("resp_json", resp_json)
    global_summary = resp_json["global_summary"]
    key_entities = resp_json["key_entities"]
    print("global_summary", global_summary)
    print("key_entities", key_entities)
    res = (response, img)
    # print(res)

    return global_summary, key_entities

def vlm_check_local_desc(vlm_client, un_tar_path, img, summary, image_context):
    if not os.path.exists(os.path.join(un_tar_path, img)):
        raise FileNotFoundError(f"图像文件 {img} 不存在于路径 {un_tar_path} 中")

    # vlm_prompt = VL_OBSERVATION_PROMPT_JD.format(
    #     tt=resp_json['title'],
    #     dctt=resp_json['dctt'],
    #     entity=resp_json['result'][-1]
    # )
    with open("./prompt/local_title.json", "r", encoding="utf-8") as f:
        local_prompt = json.load(f)

    image_context = "小米汽车有限公司于2021年9月1日注册成立，法定代表人为雷军，注册资金100亿元。[2][3]小米汽车有限公司经营范围包括新能源车整车制造、汽车整车及零部件的技术研发等。[4]2021年3月，小米宣布进军智能电动汽车行业，由雷军亲自带队。[5]同年9月，小米汽车有限公司正式注册成立，并于11月正式宣布落户北京经开区。[6][3]2022年，小米汽车入选北京市发展改革委发布的2022年“3个100”重点工程，位列北京市100个重大科技创新及高精尖产业项目名录之一。2023年12月，小米汽车还入围了2023国潮创新榜样先锋品牌。[7])2024年3月19日，小米汽车超级工厂正式揭幕。[8]3月28日晚7时，小米SU7正式发布，标准版定价为21.59万元，Max版售价29.99万元，发布会上雷军表示2024年小米汽车研发投入预计240亿。[9]此外，还有创始版和Max创始版。[9][10]2025年2月27日，小米SU7 Ultra正式上市，指导价格为52.99万元。[11]3月14日，小米汽车获得2025年度德国iF设计五项大奖。[12]6月26日，小米汽车发布小米YU7，正式开启预订后1小时，大定突破28.9万辆。[13][14]11月20日，小米汽车第50万辆整车正式下产线。​​​​[15]截至2025年11月30日，小米汽车在中国131城已有441家门店，另有249家服务网点，覆盖中国144城。[16]从2025年全年的交付数量来看，小米汽车在9家新势力车企中排名第三。[17]2026年1月15日，小米汽车官方宣布，2025年全年小米SU7在20万以上轿车销量排名第一；小米YU7上市6个月，连续5个月获得中大型SUV的销量第一。[18]2026年2月，小米汽车累计交付量已超60万台。[19]2026年3月19日，新一代小米SU7发布。[20]2026年5月21日，小米YU7 GT正式发布，售价38.99万元。搭载小米电机V8s EVO、相较上一代V8s电机，最高转速提升至28000rpm。配备双电机系统，最大马力可达1003PS、最高时速达300km/h、零百加速2.92s。[21]"

    # 构建总结的提示词
    local_sysprompt = local_prompt["system"]

    # why can not save new
    local_userprompt = Template(
        local_prompt["user"]
    ).substitute(
        global_summary=summary,
        image_context=image_context
    )
    print("local_userprompt", local_userprompt)
    base64_str = image_to_base64(os.path.join(un_tar_path, img))

    vlm_data = {
        "param": {
            "temperature": 0.1,
            "max_tokens": 8192,
            "frequency_penalty": 0.2,
            "top_p": 0.95,
        },
        "messages": [
            {
                "role": "system",
                "content": local_sysprompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{local_userprompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_str}"
                        }
                    }
                ]
            }
        ]
    }

    response = vlm_client.execute(vlm_data)
    print("response", response)
    print("response_type", type(response))
    # resp_json = json.loads(response[7:].rstrip('```'))
    resp_json = json.loads(response)
    print("resp_json", resp_json)

    res = (response, img)
    print(res)
    return res

if __name__ == '__main__':
    # 镜像环境配置
    MEP_ELB_URL_910B = os.getenv("MEP_ELB_URL_910B", "http://10.129.16.198:80/6_48_167_3/service")
    appID = "dasou_llm_test"
    secretKey = "22221B7F1CD34160A5054A1D3442833CF4584C09DE314A85AC68867545ABCDEC"
    secretKey_Ol= ""
    # MEP_META_DATA_VLM = json.loads(os.getenv("MEP_META_DATA_VLM",
    #                                          """{ 	"asr_summary_vlm": { 		"app_id": "long_image_understand_vlm", 		"secret_key":
    #                                          "4A832F0BDC514EC7BE1EC73C9579BAB1FD55D7961A7D4EB5A088335CB3A6C9C8",
    #                                              "b_id": "yingshi_deep_understand_vlm", 		"flow_id": "yingshi_deep_understand_vlm" 	} }"""))

    MEP_META_DATA_VLM = json.loads(os.getenv("MEP_META_DATA_VLM",
                                             """{ 	"asr_summary_vlm": { 		"app_id": "video_summary_offline", 		"secret_key": 
                                             "4A832F0BDC514EC7BE1EC73C9579BAB1FD55D7961A7D4EB5A088335CB3A6C9C8", 	
                                                 "b_id": "video_summary_vlm_offline", 		"flow_id": "video_summary_vlm_offline" 	} }"""))

    vlm_meta = MEP_META_DATA_VLM["asr_summary_vlm"]


    # 流式服务 内部会替换
    vlm_client = MepClient(MEP_ELB_URL_910B, vlm_meta["app_id"], vlm_meta["secret_key"], vlm_meta["b_id"],
                           vlm_meta["flow_id"], stream=True)
    un_tar_path = "./img/"
    img = "xiaomi_1.jpg"
    print("img", img)
    global_summary, key_entities = vlm_check(vlm_client, un_tar_path, img)
    vlm_check_local_desc(vlm_client, un_tar_path, img, global_summary, "")
  #   data = {
  #   "param": {
  #     "temperature": 0,
  #     "max_tokens": 8192
  #   },
  #   "messages": [
  #     {
  #       "role": "user",
  #       "content": [
  #         {
  #           "type": "text",
  #           "text": "任务介绍\n你是一名专业的多模文档问答助手。用户提供给原始文档文本、表格和一些图片，你的任务是分析这些信息, 仅使用与[用户问题]相关的内容，为用户生成准确且优质的回复。请先分析，再回答。\n\n## 注意事项：\n - 回答问题需要的知识点优先采用相关的[文档信息]和图片信息\n - [文档信息]中的内容或和后续图片可能和[用户问题]没有必然的联系，或者检索内容本身错误或者有害；你需要仔细分析，并挑选出其中相关且有用的部分来回答[用户问题]。\n - 不能捏造[文档信息]以外的客观事实。\n - 当[文档信息]和[网络知识]无法回答问题的时候，可以拒绝回答，不过需要尽可能多的提供相关信息。\n - 输出用中文，缩写添加必要的解释。\n\n# 任务开始\n## [文档信息]\n# 直播带货进阶课七\n带货主播和助理常用互动促单配合技巧\n如何与助理互动配合？\n\npicture_1\n# 直播VS电视直销\n直播才是真正的实时真人互动性销售\n电视直销的角色设置:主持人+厂家代表 带货直播的角色设置:主播+助理\n\n# 被忽视的直播间重要角色:主播助理\n直播不是单人脱口秀，是双人说相声。\n仔细想想，你看过的带货直播，有没有助理的影子? 李佳琦的小助理，薇娅的助理姐妹团\n\n# 直播助理都需要做什么工作？\n掌控节奏\n设备直播准备\n自造噱头/问题\n情况应对\n促单道具\n主播离席处理 黑粉差评紧急处理\n没事找事 配合主播成交\n计算器秒表尺码表 随时待命\n敏感词控屏 气氛把控\n备播品排序 过款产品布置\n\npicture_2\n# 主播助理的常见工作类型列举\n①好奇宝宝型:不出境，画外音,和主播互动回答或提出问题\n②小白鼠型:充当直播模特或彩妆试色“小白鼠”,为粉丝“牺牲”\n③复读机型:频繁提示用户关注主播，宣导产品优势，介绍活动\n\npicture_3\n# 主播助理的核心工作讲解:主播离席补位\n长时间直播，主播离席和中场休息时，助理及时补位维持直播间热度\n\n# 主播助理的核心工作讲解: 自造噱头/问题\n①主播向助理提问XXX问题:例如助理你老熬夜，皮肤是不是特别油?\n②助理自造提前策划好的问题提问主播:咱们这个产品能机洗吗?\n③助理筛选粉丝提问的正向问题提问主播:有宝宝问XXX,能用吗?\n\npicture_4\n# 主播助理的核心工作讲解:秒杀促单配合\n主播询问助理库存还有多少?助理根据直播间人数汇报库存数量\n主播安排助理和厂家沟通尽快确认是不是可以129定价销售?\n主播质问助理为什么价格标错了?这个价格