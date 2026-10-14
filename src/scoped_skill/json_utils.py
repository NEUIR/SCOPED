import json
import re


def extract_json(text):
    """从文本中提取第一个平衡的 JSON 对象;失败返回 None。"""
    if not text:
        return None
    # 去掉 ```json ``` 围栏
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    frag = text[start:i + 1]
                    try:
                        return json.loads(frag)
                    except json.JSONDecodeError:
                        try:
                            return json.loads(frag.replace("'", '"'))
                        except json.JSONDecodeError:
                            return None
    return None

def repair_json(text):
    """解析可能被 max_tokens 截断的 JSON:先正常解析,失败则补齐闭合括号抢救。"""
    if not text:
        return None
    normal = extract_json(text)
    if normal is not None:
        return normal
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        return None
    s = text[start:]
    last_brace = s.rfind("}")
    if last_brace == -1:
        return None
    s = s[:last_brace + 1]
    depth_curly = depth_square = 0
    in_str = esc = False
    for c in s:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth_curly += 1
            elif c == "}":
                depth_curly -= 1
            elif c == "[":
                depth_square += 1
            elif c == "]":
                depth_square -= 1
    s = re.sub(r",\s*$", "", s.rstrip())
    s = s + "]" * max(0, depth_square) + "}" * max(0, depth_curly)
    for candidate in (s, s.replace("'", '"')):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
