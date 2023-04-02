import logging
import os
from typing import List

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from experiment_infrastructure.attacks.attack import Attack
from ml.datasets.dataset import Dataset
from ml.datasets.partitioner import DataPartitioner, DirichletDataPartitioner


class KDDCup99Dataset(Dataset):
    """
    Dataset containing the following classes and features:

    back,buffer_overflow,ftp_write,guess_passwd,imap,ipsweep,land,loadmodule,multihop,neptune,nmap,normal,perl,phf,pod,portsweep,rootkit,satan,smurf,spy,teardrop,warezclient,warezmaster.
    duration: continuous.
    protocol_type: symbolic.
    service: symbolic.
    flag: symbolic.
    src_bytes: continuous.
    dst_bytes: continuous.
    land: symbolic.
    wrong_fragment: continuous.
    urgent: continuous.
    hot: continuous.
    num_failed_logins: continuous.
    logged_in: symbolic.
    num_compromised: continuous.
    root_shell: continuous.
    su_attempted: continuous.
    num_root: continuous.
    num_file_creations: continuous.
    num_shells: continuous.
    num_access_files: continuous.
    num_outbound_cmds: continuous.
    is_host_login: symbolic.
    is_guest_login: symbolic.
    count: continuous.
    srv_count: continuous.
    serror_rate: continuous.
    srv_serror_rate: continuous.
    rerror_rate: continuous.
    srv_rerror_rate: continuous.
    same_srv_rate: continuous.
    diff_srv_rate: continuous.
    srv_diff_host_rate: continuous.
    dst_host_count: continuous.
    dst_host_srv_count: continuous.
    dst_host_same_srv_rate: continuous.
    dst_host_diff_srv_rate: continuous.
    dst_host_same_src_port_rate: continuous.
    dst_host_srv_diff_host_rate: continuous.
    dst_host_serror_rate: continuous.
    dst_host_srv_serror_rate: continuous.
    dst_host_rerror_rate: continuous.
    dst_host_srv_rerror_rate: continuous.
    """

    # label_type = ['normal', 'ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan', 'apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf',
    #                'teardrop', 'udpstorm', 'buffer_overflow', 'httptunnel', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack',
    #                'xterm', 'ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack',
    #                'snmpguess', 'spy', 'warezclient', 'warezmaster', 'worm', 'xlock', 'xsnoop']

    label_type = [['normal'],
                  ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'],
                  ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf',
                   'teardrop', 'udpstorm'],
                  ['buffer_overflow', 'httptunnel', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack',
                   'xterm'],
                  ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack',
                   'snmpguess', 'spy', 'warezclient', 'warezmaster', 'worm', 'xlock', 'xsnoop']]

    encoders = [
        None,
        ['icmp', 'tcp', 'udp'],
        ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime',
         'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs',
         'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames',
         'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap',
         'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name',
         'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u',
         'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i',
         'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc',
         'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i',
         'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50',
         'icmp'],
        ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH'],
        None,
        None,
        None,  # dict(),
        None,
        None,
        None,
        None,
        None,  # dict(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,  # dict(),
        None,  # dict(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def read_file(self, filename):
        with open(filename, 'r') as f:
            data = [line.rstrip() for line in f]
        return data

    def get_all_data(self):
        data = self.read_file(os.path.join(self.DEFAULT_DATA_DIR, 'train/KDDTrain+.txt'))

        all_data = list()
        class_encoder = dict()

        for i in range(len(data)):
            features = data[i].split(",")[:-2]
            converted_features: List[float] = list()
            label = data[i].split(",")[-2]
            for j in range(len(features)):
                feat = features[j]
                if self.encoders[j] is None:
                    converted_features.append(float(feat))
                else:
                    converted_features.append(self.encoders[j].index(feat))

            # if label not in class_encoder:
            #     class_encoder[label] = len(class_encoder)
            y = -1
            for labels in self.label_type:
                if labels.count(label) > 0:
                    y = self.label_type.index(labels)

            if y == -1:
                raise ValueError("Label not found" + str(label))

            all_data.append((torch.Tensor(converted_features), y))

        data = self.read_file(os.path.join(self.DEFAULT_DATA_DIR, 'test/KDDTest+.txt'))

        all_test_data = list()
        print(class_encoder)

        for i in range(len(data)):
            features = data[i].split(",")[:-2]
            converted_features: List[float] = list()
            label = data[i].split(",")[-2]
            for j in range(len(features)):
                feat = features[j]
                if self.encoders[j] is None:
                    converted_features.append(float(feat))
                else:
                    converted_features.append(self.encoders[j].index(feat))

            y = -1
            for labels in self.label_type:
                if labels.count(label) > 0:
                    y = self.label_type.index(labels)

            if y == -1:
                raise ValueError("Label not found" + str(label))
            all_test_data.append((torch.Tensor(converted_features), y))

        return all_data, all_test_data

    def all_training_data(self, batch_size=32, shuffle=False):
        return DataLoader(torchvision.datasets.MNIST(
            root=self.DEFAULT_DATA_DIR + '/train', train=True, download=True, transform=ToTensor(),
        ), batch_size=batch_size, shuffle=shuffle)

    def get_peer_dataset(self, peer_id: int, total_peers: int, non_iid=False, sizes=None, batch_size=8, shuffle=False,
                         sybil_data_transformer: Attack = None):
        self.logger.info(f"Initializing dataset of size {1.0 / total_peers} for peer {peer_id}. Non-IID: {non_iid}")
        if sizes is None:
            sizes = [1.0 / total_peers for _ in range(total_peers)]
        data = torchvision.datasets.MNIST(
            root=self.DEFAULT_DATA_DIR + '/train', train=True, download=True, transform=ToTensor(),
        )
        if not non_iid:
            train_set = DataPartitioner(data, sizes).use(peer_id)
        else:
            train_set = DirichletDataPartitioner(
                data, sizes
            ).use(peer_id)
        if sybil_data_transformer is not None:
            train_data = {key: [] for key in range(10)}
            for x, y in data:
                train_data[y].append(x)
            train_set = sybil_data_transformer.transform_data(train_set, train_data, sizes, peer_id)

        return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

    def all_test_data(self, batch_size=32, shuffle=False):
        return DataLoader(torchvision.datasets.MNIST(
            root=self.DEFAULT_DATA_DIR + '/test', train=False, download=True, transform=ToTensor()
        ), batch_size=batch_size, shuffle=shuffle)
