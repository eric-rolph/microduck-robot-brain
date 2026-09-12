"""
Unit tests for PollenRobotdClient JSON-RPC protocol formatting.
"""

from unittest.mock import MagicMock
from microduck_brain.pollen_bridge import PollenRobotdClient


def test_pollen_client_notification_formatting():
    client = PollenRobotdClient()
    mock_sock = MagicMock()
    client.sock = mock_sock

    client.move(vx=0.2, vy=0.0, vyaw=0.4)
    assert mock_sock.sendall.called
    sent_data = mock_sock.sendall.call_args[0][0].decode("utf-8")
    assert '"method": "robot.move"' in sent_data
    assert '"vx": 0.2' in sent_data
    assert '"vyaw": 0.4' in sent_data


def test_pollen_client_head_formatting():
    client = PollenRobotdClient()
    mock_sock = MagicMock()
    client.sock = mock_sock

    client.head(neck_pitch=0.1, head_pitch=0.2, head_yaw=0.3, head_roll=0.0)
    assert mock_sock.sendall.called
    sent_data = mock_sock.sendall.call_args[0][0].decode("utf-8")
    assert '"method": "robot.head"' in sent_data
    assert '"neck_pitch": 0.1' in sent_data


def test_pollen_client_skill_request():
    client = PollenRobotdClient()
    mock_sock = MagicMock()
    mock_sock.recv.return_value = b'{"jsonrpc":"2.0","id":1,"result":{"accepted":true}}\n'
    client.sock = mock_sock

    resp = client.do_skill("ground_pick")
    assert resp is not None
    assert resp["result"]["accepted"] is True
    sent_data = mock_sock.sendall.call_args[0][0].decode("utf-8")
    assert '"method": "robot.do"' in sent_data
    assert '"skill": "ground_pick"' in sent_data
