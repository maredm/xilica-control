import io
"""Defined according to https://audiovias.com/descarga/534/xilica-manuales/10874/xilica-xconsole-communication-protocol.pdf"""
import socket
import threading
from time import sleep
import enum
import numpy as np
import copy
import dataclasses


PORT = 10001
MTU = 256

_START_DELIMITER = b'\x01'
_END_DELIMITER = b'\x02'
_DEFAULT_SENDER = b'\x7F'
_PROCESS = b'\x1F'


_CHARACTER_SET = ("_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvw"
                  "xyz-<>?,./~!@#$%^&*()=+';")
def cord(char):
  return _CHARACTER_SET.index(char)
def cchr(value):
  return _CHARACTER_SET[value]

class Model(enum.Enum):
  XP4080 = enum.auto()


class Mode(enum.Enum):
  READ = b'\x52'
  WRITE = b'\x57'


class Direction(enum.Enum):
  INPUT = 0
  OUTPUT = 1


class FilterShape(enum.Enum):
  PEAKING = 0
  LOW_SHELF = 1
  HIGH_SHELF = 2
  AP_1 = 3
  AP_2 = 4

class FilterType(enum.Enum):
  OFF = 0
  BUTTERWORTH = 1
  LINKWITZ_RILEY = 2
  BESSEL = 3

class Command(enum.Enum):
  METER = b'MTR0', float
  MUTE = b'MUT0', bool, 0, 1, 0, 1
  MIX_GAIN = b'MIC0', float, 0, 15, 0, 45, 3
  SIGNAL_LEVEL = b'LVL0', float, 0, 220, -40, 15, 0.25
  SIGNAL_POLARITY = b'POL0', int, 0, 1
  SIGNAL_DELAY = b'DLY3', float, 0, 62400, 0, 650, 1/96
  EQ_TYPE = b'EQT0', FilterShape, 0, 4
  EQ_FREQUENCY = b'EQF0', float, 0, 29980, 20, 30000
  EQ_BANDWIDTH = b'EQB0', float, 0, 359, 0.02, 3.61, 0.01
  EQ_LEVEL = b'EQL0', float, 0, 180, -30, 15, 0.25
  EQ_BYPASS = b'EQb0', bool, 0, 1, 0, 1
  GEQ_LEVEL = b'EQL1', float, 0, 180, -30, 15, 0.25
  GEQ_BYPASS = b'EQb1', bool, 0, 1, 0, 1
  CROSSOVER_TYPE = b'XRT0', FilterType, 0, 3
  CROSSOVER_FREQUENCY = b'XRF0', float, 0, 29980, 20, 30000
  CROSSOVER_SLOPE = b'XRS0', int, 0, 7, 6, 48, 6
  FIR_TYPE = b'XFE0', int, 0, 1, 0, 1
  FIR_FREQUENCY = b'XFF0', float, 0, 29980, 20, 30000
  COMPRESSOR_THRESHOLD = b'CMT0', float, 0, 80, -20, 20, 0.5
  COMPRESSOR_ATTACK = b'CMA0', float, 0, 106, 0.0003, 0.1, 0.1
  COMPRESSOR_RELEASE = b'CMR0', float, 0, 4
  COMPRESSOR_RATIO = b'CMX0', float, 0, 39, 1, 40
  LIMITER_THRESHOLD = b'LMT0', float, 0, 80, -20, 20, 0.5
  LIMITER_ATTACK = b'LMA0', float, 0, 106, 0.0003, 0.1, 0.1
  LIMITER_RELEASE = b'LMR0', float, 0, 4
  MIXER = b'MIX0', float, 0, 161, -40, 0, 0.25
  CHANNEL_NAME = b'NCH0', str, 0, 84, cord('_'), cord(';')
  DOWN_SYNC = b'%SD0'
  UP_SYNC = b'%SU0'
  LOCK_PASSWORD = b'%LP0', int
  LOCK_KEY = b'%LK0', int
  CHANNEL_IN_NUMBER = b'%nI0',
  CHANNEL_OUT_NUMBER = b'%nO0'
  COMPANY = b'%NC0', str
  SAMPLING_FREQUENCY = b'%nS0', str
  PRODUCT_NAME = b'%NP0', str, 0, 84, cord('_'), cord(';')
  VERSION = b'%NV0', str
  DEVICE_NAME = b'%ND0', str, 0, 84, cord('_'), cord(';')
  DEVICE_NUMBER = b'%nD0', int
  N_NUMBER = b'%nN0', int
  PROGRAM_NUMBER = b'%Pn0', int
  PROGRAM_ORDER = b'%PO0', int
  PROGRAM_NAME = b'%PN0', str, 0, 84, cord('_'), cord(';')
  PROGRAM_RECALL = b'%PR0'
  PROGRAM_STORE = b'%PS0'
  PROGRAM_DOWNLOAD = b'%PD0'
  PROGRAM_UPLOAD = b'%PU0'
  RESET = b'%RS0'
  ETHERNET_INFO = b'%EN0', int
  XPANEL_INFO = b'%XP0', str
  LEVEL_UP = b'#LVL'  # Column 0 for up, column 1 for down. 0.25 dB steps.

  def __new__(cls, value,
              data_type:type|list|None=None,
              data_min:int|None=None,
              data_max:int|None=None,
              value_min:int|None=None,
              value_max:int|None=None,
              value_step:int=1):
    entry = object.__new__(cls)
    entry._value_ = value  # set the value, and the extra attribute
    entry.type = data_type
    entry.data_min = data_min
    entry.data_max = data_max
    entry.value_min = value_min
    entry.value_max = value_max
    entry.value_step = value_step

    return entry
  def actual_value(self, data: int):
    if self.value_min is not None and self.value_max is not None:
      return self.value_min + self.value_step * data
    else:
      return 'N/A'


class Header(enum.Enum):
  STRING_4B = 3, 4
  STRING_5B = 4, 5
  STRING_6B = 5, 6
  STRING_7B = 6, 7
  STRING_8B = 7, 8
  DEVICE = 8
  IO = 9
  CHANNEL = 10
  AUX = 11
  COLUMN = 12
  DATA_1B = 16
  DATA_2B = 17, 2
  DATA_3B = 18, 3
  DATA_4B = 19, 4
  DATA_5B = 20, 5
  DATA_6B = 21, 6
  DATA_7B = 22, 7
  DATA_8B = 23, 8
  DATA_9B = 24, 9
  PROCESS = 31, 10

  def __new__(cls, value,
              length:int=1):
    entry = object.__new__(cls)
    entry._value_ = value  # set the value, and the extra attribute
    entry.length = length

    return entry

  @property
  def value(self):
    return super().value.to_bytes(1, 'big')


def _checksum(buffer):
  buffer.seek(0)
  return (sum(buffer.read()) % 256 % 96 + 32).to_bytes(1, 'big')


def _as_value(value):
  return (32 + value).to_bytes(1, 'big')


@dataclasses.dataclass
class Request():
  command: Command
  device: int = 0
  column: int|None = None
  aux: int|None = None
  io: int|None = None
  channel: int|None = None
  data: int|None = None
  data_header: Header = None
  processed: True = False

  def __str__(self):
    ignore = ['command', 'device', 'processed']
    output = f'{self.command.name} - device {self.device}'
    for x in dir(self):
      if not x.startswith('__') and x not in ignore:
        value = getattr(self, x)
        if value is not None:
          output += f' {x} {getattr(self, x)}'
    return output


@dataclasses.dataclass
class Channel:
  name: str = ""


@dataclasses.dataclass
class Program:
  name: str = ""


@dataclasses.dataclass
class DSP:
  programs: dict[Channel] = dataclasses.field(default_factory=dict)
  channels: dict[Program] = dataclasses.field(default_factory=dict)
  metadata: dict[str] = dataclasses.field(default_factory=dict)
  

class Client:
  def __init__(
        self,
        endpoint: str,
        device_id: int,
        timeout:float=5,
        model:Model=Model.XP4080
    ):
    self.id = device_id
    self.endpoint = endpoint
    self.connected =  False

  def connect(self):
    try:
      self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self._socket.settimeout(1)
      self._socket.connect((self.endpoint, PORT))
    except TimeoutError as e: 
      raise TimeoutError(f'Unable to connect to {self.endpoint}:{PORT}') from e
    self.connected = True
    self.dsp = DSP()

  def _format_parameter(self, name, value):
    return b'\x00'

  def prepare_packet(self, command: Command, parameters, mode=Mode.WRITE):
    """Returns a packet of the correct format, based on the current instruction.

    Format:
      <01><READ/WRITE><SENDER><HEADER><VALUE>...<PROCESS><CHECKSUM><02>

      The <HEADER><VALUE> combo can be repeated as long as the total length is
      kept below 256 Bytes.

      The <PROCESS> command can be inserted multiple times to synchronize
      different actions.

    Args:
      command: Command enum object to send.
      parameters: Tuple pairs of Header enum objects and integer values.

    Raises:
      ValueError: if the packet is larger than the MTU of 256 Bytes.

    """
    buffer = io.BytesIO()
    buffer.write(_START_DELIMITER)
    buffer.write(mode.value)
    buffer.write(_DEFAULT_SENDER)

    # Command
    buffer.write(Header.STRING_4B.value)
    buffer.write(command.value)

    buffer.write(Header.DEVICE.value)
    buffer.write(_as_value(self.id))

    for name, value in parameters:
      if name is Header.PROCESS:
        buffer.write(name.value)
        continue
      if isinstance(value, enum.Enum):
        value = value.value

      if 'DATA' in name.name:
        if command.data_min is not None:
          if command.data_min > value:
            raise ValueError(
                f'Value for parameter {command.name} exceeded the minimum value'
                f' limit of {command.data_min} (was {value}).')
        if command.data_max is not None:
          if command.data_max < value:
            raise ValueError(
                f'Value for parameter {command.name} exceeded the maximum value'
                f' limit of {command.data_max} (was {value}).')

      buffer.write(name.value)
      buffer.write(_as_value(value))

    buffer.write(_checksum(buffer))
    buffer.write(_END_DELIMITER)
    buffer.seek(0)

    return buffer  # self.send(buffer)

  def mute(self, channel: int, direction: Direction):
    command = Command.MUTE
    parameters = (
        (Header.IO, direction),
        (Header.CHANNEL, channel),
        (Header.DATA_1B, True),
        (Header.PROCESS, None),
    )

    return self.prepare_packet(command, parameters)

  def set_signal_level(self, level: int, channel: int):
    command = Command.SIGNAL_LEVEL
    parameters = (
        (Header.CHANNEL, 2),
        (Header.DATA_1B, level),
    )

    packet = self.prepare_packet(command, parameters)
    return packet  # self.send(packet)

  def set_passwcord(self, value: str):
    command = Command.LOCK_PASSWORD
    parameters = (
        (Header.AUX, 0),
        (Header.DATA_1B, cord('A')),
        (Header.PROCESS, None),
        (Header.AUX, 1),
        (Header.DATA_1B, cord('B')),
        (Header.AUX, 2),
        (Header.DATA_1B, cord('C')),
        (Header.AUX, 3),
        (Header.DATA_1B, cord('D')),
    )

    packet = self.prepare_packet(command, parameters)
    return packet  # self.send(packet)

  def recall_program(self, id: int):
    command = Command.PROGRAM_RECALL
    parameters = (
        (Header.DATA_1B, id),
        (Header.PROCESS, None),
    )

    packet = self.prepare_packet(command, parameters)
    return packet  # self.send(packet)

  def upload_program(self, id: int):
    command = Command.PROGRAM_UPLOAD
    parameters = (
        (Header.DATA_1B, 1),
        (Header.PROCESS, None),
    )

    packet = self.prepare_packet(command, parameters)
    return packet  # self.send(packet)

  def upload(self, on: bool):
    command = Command.PROGRAM_UPLOAD
    parameters = (
        (Header.DATA_1B, 1 if on else 0),
        (Header.PROCESS, None),
    )

    packet = self.prepare_packet(command, parameters)
    return packet  # self.send(packet)

  def upsync(self):
    command = Command.UP_SYNC
    parameters = (
        (Header.PROCESS, None),
    )

    packet = self.prepare_packet(command, parameters, mode=Mode.READ)
    print(packet.read())
    packet.seek(0)
    return packet  # self.send(packet)

  def send(self, data: bytes):
    if len(data) > MTU:
      raise ValueError(
          f'Packet length is limited to 256 B (was {len(data)}B). Aborting!')
    self._socket.send(data)

    print("Listening to receiver at", self.endpoint)

    # Wait for responses and call parser each time there is a response.
    replies = []
    while True:
      try:
        data = self._socket.recv(1024)
      except TimeoutError:
        break
      replies.extend(decode(data))

    for r in replies:
      try:
        diff_character = [Command.PROGRAM_NAME, Command.CHANNEL_NAME, Command.DEVICE_NAME]
        if r.command.type == str:
          if r.command in diff_character:
            chr_command = cchr
          else:
            chr_command = chr
            #r.data = r.data + 32
          if r.channel is not None:
            if not self.dsp.metadata.get(r.command.name.lower()):
              self.dsp.metadata[r.command.name.lower()] = dict()
            if r.io is not None:
              if self.dsp.metadata[r.command.name.lower()].get(r.io) is None:
                self.dsp.metadata[r.command.name.lower()][r.io] = dict()
              try:
                if not self.dsp.metadata[r.command.name.lower()][r.io].get(r.channel):
                  self.dsp.metadata[r.command.name.lower()][r.io][r.channel] = ''
                self.dsp.metadata[r.command.name.lower()][r.io][r.channel] += chr_command(r.data)
              except:
                pass
            else:
              try:
                if not self.dsp.metadata[r.command.name.lower()].get(r.channel):
                  self.dsp.metadata[r.command.name.lower()][r.channel] = ''
                self.dsp.metadata[r.command.name.lower()][r.channel] += chr_command(r.data)
              except:
                pass
          else:
            if not self.dsp.metadata.get(r.command.name.lower()):
              self.dsp.metadata[r.command.name.lower()] = ''
            self.dsp.metadata[r.command.name.lower()] += chr_command(r.data)
        if r.command == Command.PROGRAM_NAME:
            if not self.dsp.programs.get(r.channel):
              self.dsp.programs[r.channel] = Program()
            self.dsp.programs[r.channel].name += chr_command(r.data)
        if r.command == Command.CHANNEL_NAME:
            if not self.dsp.channels.get(r.channel + r.io*4):
              self.dsp.channels[r.channel + r.io*4] = Channel()
            self.dsp.channels[r.channel + r.io*4].name += chr_command(r.data)
      except:
        pass  
      if r.command.type in [int, float, bool]:
        value = (r.data - (r.command.data_min or 0)) * r.command.value_step + (r.command.value_min or 0)
        if r.channel is not None:
          if not self.dsp.metadata.get(r.command.name.lower()):
              self.dsp.metadata[r.command.name.lower()] = dict()
          if r.io is not None:
            if self.dsp.metadata[r.command.name.lower()].get(r.io) is None:
              self.dsp.metadata[r.command.name.lower()][r.io] = dict()
            try:
              if not self.dsp.metadata[r.command.name.lower()][r.io].get(r.channel):
                self.dsp.metadata[r.command.name.lower()][r.io][r.channel] = []
              self.dsp.metadata[r.command.name.lower()][r.io][r.channel].append(value)
            except:
              pass
          else:
            try:
              if not self.dsp.metadata[r.command.name.lower()].get(r.channel):
                self.dsp.metadata[r.command.name.lower()][r.channel] = []
              self.dsp.metadata[r.command.name.lower()][r.channel].append(value)
            except:
              pass
        else:
          if not self.dsp.metadata.get(r.command.name.lower()):
            self.dsp.metadata[r.command.name.lower()] = []
          self.dsp.metadata[r.command.name.lower()].append(value)
      if r.command.type in [FilterShape, FilterType]:
        value = (r.data - (r.command.data_min or 0)) * r.command.value_step + (r.command.value_min or 0)
        print(value)
        print(r.command.type(int(value)).name)
        if r.channel is not None:
          if not self.dsp.metadata.get(r.command.name.lower()):
              self.dsp.metadata[r.command.name.lower()] = dict()
          if r.io is not None:
            if self.dsp.metadata[r.command.name.lower()].get(r.io) is None:
              self.dsp.metadata[r.command.name.lower()][r.io] = dict()
            try:
              if not self.dsp.metadata[r.command.name.lower()][r.io].get(r.channel):
                self.dsp.metadata[r.command.name.lower()][r.io][r.channel] = []
              self.dsp.metadata[r.command.name.lower()][r.io][r.channel].append(r.command.type(value).name)
            except:
              pass
          else:
            try:
              if not self.dsp.metadata[r.command.name.lower()].get(r.channel):
                self.dsp.metadata[r.command.name.lower()][r.channel] = []
              self.dsp.metadata[r.command.name.lower()][r.channel].append(r.command.type(value).name)
            except:
              pass
        else:
          if not self.dsp.metadata.get(r.command.name.lower()):
            self.dsp.metadata[r.command.name.lower()] = []
          self.dsp.metadata[r.command.name.lower()].append(r.command.type(value).name)
    import pprint
    pprint.pprint(self.dsp.metadata)
    filters = []
    frequency = self.dsp.metadata.get('eq_frequency')[1][0]
    gain = self.dsp.metadata.get('eq_level')[1][0]
    bypass = self.dsp.metadata.get('eq_bypass')[1][0]
    bandwidth = self.dsp.metadata.get('eq_bandwidth')[1][0]
    eq_type = self.dsp.metadata.get('eq_type')[1][0]

    for i, fc in enumerate(frequency):
      if not bypass[i]:
        print('Filter at', fc, 'with', gain[i], 1.41/bandwidth[i], eq_type[i].lower())
        filters.append(IIRfilter(G=gain[i], fc=fc, rate=fs, Q=1.41/bandwidth[i], filter_type=eq_type[i].lower()))

    
    frequency = self.dsp.metadata.get('crossover_frequency')[1][0]
    slope = self.dsp.metadata.get('crossover_slope')[1][0]
    eq_type = self.dsp.metadata.get('crossover_type')[1][0]

    # https://www.hxaudiolab.com/uploads/2/5/5/3/25532092/cascading_biquads_to_create_even-order_highlow_pass_filters_2.pdf
    for i, fc in enumerate(frequency):
      filter_type = 'high_pass' if i == 0 else 'low_pass'
      if eq_type[i] == 'LINKWITZ_RILEY' and slope[i] == 12:
        print('Filter at', fc, 'with slope', slope[i], 'dB', eq_type[i].lower())
        filters.append(IIRfilter(G=gain[i], fc=fc, rate=fs, Q=0.5, filter_type=filter_type))
      if eq_type[i] == 'LINKWITZ_RILEY' and slope[i] == 24:
        print('Filter at', fc, 'with slope', slope[i], 'dB', eq_type[i].lower())
        filters.append(IIRfilter(G=gain[i], fc=fc, rate=fs, Q=0.707, filter_type=filter_type))
        filters.append(IIRfilter(G=gain[i], fc=fc, rate=fs, Q=0.707 , filter_type=filter_type))
      if eq_type[i] == 'LINKWITZ_RILEY' and slope[i] == 48:
        print('Filter at', fc, 'with slope', slope[i], 'dB', eq_type[i].lower())
        filters.append(IIRfilter(G=gain[i], fc=fc, rate=fs, Q=0.54, filter_type=filter_type))
        filters.append(IIRfilter(G=gain[i], fc=fc, rate=fs, Q=0.54, filter_type=filter_type))
        filters.append(IIRfilter(G=gain[i], fc=fc, rate=fs, Q=1.31, filter_type=filter_type))
        filters.append(IIRfilter(G=gain[i], fc=fc, rate=fs, Q=1.31, filter_type=filter_type))
    save_filters(filters)

    return replies

import matplotlib.pyplot as plt
import matplotlib.colors as c
import matplotlib
matplotlib.use('agg')

import numpy as np
import scipy
from scipy.signal import freqz
from pyloudnorm import IIRfilter


fs=48000

def db(value: np.ndarray) -> np.ndarray:
  return 20 * np.log10(value)

def save_filters(filters: list[IIRfilter]):
  fig = plt.figure(figsize=(12, 3.5))
  fig, ax1 = plt.subplots(figsize=(12, 3.5))

  sum = np.ones((48000)).astype(np.complex128)
  for index, filter in enumerate(filters):
    w, h = scipy.signal.freqz(filter.b, filter.a, 48000, fs=fs)
    ax1.semilogx(w, db(abs(h)), linewidth=0.75)
    sum *= h
    ax1.fill_between(w, db(h), 0, alpha=0.2) 

  ax1.semilogx(w, db(abs(sum)), color=c.CSS4_COLORS['darkslategray'])
  ax1.set_xlim(20, 20000)
  ax1.set_ylim(-30, 20)
  ax1.set_xticks([20, 100, 200, 1000, 2000, 10000, 20000])
  ax1.grid()
  fig.savefig('plots.png')

def from_value(value:bytes):
  sum = 0
  for i, v in enumerate(value):
    sum += (v - 32 % 96) * 96 ** (len(value)-i-1)
  return sum

def decode(data):
  # Prepare buffer
  buffer = io.BytesIO()
  length = len(data)
  buffer.write(data)
  buffer.seek(0)

  loaded_data = []
  #for i in data:
  #  print('<', hex(i)[2:].rjust(2, '0').title(), end='>', sep='')

  delimiter = buffer.read(1)
  if delimiter != _START_DELIMITER:
    print(f'Invalid delimiter {delimiter}')
    return []

  mode = buffer.read(1)
  if mode != Mode.WRITE.value:
    print(f'Decoder only supports write packets (why would a it read?)')
    return []

  dd = from_value(buffer.read(1))
  if 0 < dd < 32:
    buffer.read(1)
    dd = from_value(buffer.read(1))
    if dd != 95:
      buffer.seek(3)
  #print(f'Reply from device {dd}')
  command_length = int.from_bytes(buffer.read(1), 'big') + 1
  command = buffer.read(command_length)

  try:
    command = Command(command)
    loaded_data = [Request(command)]
    #print(f'Command {command.name!r}')
  except ValueError:
    #print(command)
    return []

  loaded_data[-1].command = command
  #print(command)
  while buffer.tell() < length-2:
    if loaded_data[-1].processed:
      loaded_data.append(copy.deepcopy(loaded_data[-1]))
      loaded_data[-1].processed = False
    header_ = int.from_bytes(buffer.read(1), 'big')
    if header_ > 96:
      print(header_)

    try:
      command_ = Header(header_)
      if command_ is Header.PROCESS:
        loaded_data[-1].processed = True
        continue
      if 'STRING' in command_.name:
        print('wow>!', command, command_.length)
      if 'DATA' in command_.name:
        loaded_data[-1].data = from_value(buffer.read(command_.length))
        loaded_data[-1].data_header = command_
      else:
        setattr(loaded_data[-1], command_.name.lower(), from_value(buffer.read(1)))
    except ValueError:
      pass
        
  #for i in data:
    #print('<', hex(i)[2:].rjust(2, '0').title(), end='>', sep='')
  #print()
  
  return loaded_data