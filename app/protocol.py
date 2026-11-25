"""上位机通信协议骨架。

帧格式请阅读仓库根目录的 ``PROTOCOL.md``。本文件中的两个 TODO 是通信
模块的核心考核点：封包和流式拆包。请勿假设一次 ``recv`` 恰好得到一帧。
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List


HEADER = b"\xAA\x55"
MAX_BODY_LENGTH = 64


class Command(IntEnum):
    WASTE_RESULT = 0x01
    ACK = 0x02
    ACTION_DONE = 0x03
    BIN_FULL = 0x04
    START_COMPRESS = 0x05
    HEARTBEAT = 0x06
    ERROR = 0x7F


class WasteType(IntEnum):
    HAZARDOUS = 0x01
    OTHER = 0x02
    RECYCLABLE = 0x03
    KITCHEN = 0x04


WASTE_NAMES = {
    WasteType.RECYCLABLE: "可回收垃圾",
    WasteType.KITCHEN: "厨余垃圾",
    WasteType.HAZARDOUS: "有害垃圾",
    WasteType.OTHER: "其他垃圾",
}


@dataclass(frozen=True)
class Packet:
    command: Command
    payload: bytes = b""

    def encode(self) -> bytes:
        """按协议生成完整二进制帧。"""
        body = bytes((int(self.command),)) + bytes(self.payload)
        length = len(body)
        if length > MAX_BODY_LENGTH:
            raise ValueError(
                "帧长度 {} 超过上限 {}".format(length, MAX_BODY_LENGTH)
            )
        checksum = (length + sum(body)) & 0xFF
        return HEADER + bytes((length,)) + body + bytes((checksum,))


class StreamParser:
    """把任意分片的 TCP/串口字节流增量解析为 Packet。"""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.error_count = 0

    def reset(self) -> None:
        self._buffer.clear()
        self.error_count = 0

    def feed(self, data: bytes) -> List[Packet]:
        """支持半包、粘包、噪声和错误校验后的重新同步。"""
        if data:
            self._buffer.extend(data)

        packets = []
        while True:
            index = self._buffer.find(HEADER)
            if index < 0:
                # 没有帧头时只保留可能是半个帧头的最后一个字节。
                if self._buffer and self._buffer[-1] == HEADER[0]:
                    del self._buffer[:-1]
                else:
                    self._buffer.clear()
                break
            if index > 0:
                del self._buffer[:index]
            if len(self._buffer) < 3:
                break

            length = self._buffer[2]
            if not 1 <= length <= MAX_BODY_LENGTH:
                self.error_count += 1
                del self._buffer[0]
                continue

            total = 2 + 1 + length + 1
            if len(self._buffer) < total:
                break

            body = bytes(self._buffer[3 : 3 + length])
            checksum = self._buffer[total - 1]
            if checksum != ((length + sum(body)) & 0xFF):
                self.error_count += 1
                del self._buffer[0]
                continue

            try:
                command = Command(body[0])
            except ValueError:
                self.error_count += 1
                del self._buffer[0]
                continue

            packets.append(Packet(command, body[1:]))
            del self._buffer[:total]

        return packets
