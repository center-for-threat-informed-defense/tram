"""
Reformat training data into a report export so that it can be imported into TRAM.

The target format is:
{
    "sentence-text": ["<technique-id-1>", "<technique-id-2>", "<technique-id-N>"]
}

The original format is:
    * negative_data.json - A file with sentences that have no mappings. This is a simple list of strings.
    * all_analyzed_reports.json - A file with mappings. Has the following structure:
    {
        "<attack-technique-description>": ["<sentence-1>", "<sentence-2>", "<sentence-n>"],  # OR
        "<description-1 description-2 description-N -multi>": {  # Can use key.endswith('-multi') to test
        "technique_names": [
                "description-1",
                "description-2",
                "description-N",
            ],
            "sentances": [  # Note the word sentences is misspelled as sentances
                "<sentence-1>",
                "<sentence-2>",
                "<sentence-N>"
            ]
        }
    }

The target format is defined by tram.serializers.ReportExportSerializer
"""

import json
from tram.serializers import ReportExportSerializer

outfile = 'data/training/bootstrap-training-data.json'

attack_lookup = {  # A mapping of attack descriptions to technique IDs
    'drive-by compromise': 'T1189',
    'system information discovery': 'T1082',
    'new service': 'UNKNOWN',
    'service execution': 'T1569.002',
    'command-line interface': 'T1059',  # Maps to: T1059 - Command and Scripting Interpreter
    'obfuscated files or information': 'T1027',
    'custom cryptographic protocol': 'T1573',  # Maps to: T1573 - Encrypted Channel
    'system network configuration discovery': 'T1016',
    'web shell': 'T1505.003',
    'application window discovery': 'T1010',
    'file deletion': 'T1070.004',  # Technique that became a subtechnique
    'standard application layer protocol': 'T1071',
    'web service': 'T1102',
    'exfiltration over command and control channel': 'T1041',
    'fallback channels': 'T1008',
    'bypass user account control': 'T1548.002',  # Technique that became a subtechnique
    'system time discovery': 'T1124',
    'deobfuscate/decode files or information': 'T1140',
    'disabling security tools': 'T1562.001',  # Maps to: T1562.001 - Impair Defenses: Disable or Modify Tools
    'registry run keys / startup folder': 'T1547.001',
    'remote file copy': 'T1105',  # Maps to: T1105 - Ingress Tool Transfer
    'dll search order hijacking': 'T1574.001',
    'screen capture': 'T1113',
    'file and directory discovery': 'T1083',
    'tor': 'S0183',  # Software??
    'shortcut modification': 'T1547.009',
    'remote services': 'T1021',
    'connection proxy': 'T1090',
    'data encoding': 'T1132',
    'spearphishing link': 'T1566.002',
    'spearphishing attachment': 'T1566.001',
    'arp': 'S0099',
    'user execution': 'T1204',
    'process hollowing': 'T1055.012',
    'execution through api': 'T1106',  # Maps to T1106 - Native API
    'masquerading': 'T1036',
    'code signing': 'T1553.002',
    'standard cryptographic protocol': 'T1521',
    'scripting': 'T1059',
    'remote system discovery': 'T1018',
    'credential dumping': 'T1003',
    'exploitation for client execution': 'T1203',
    'exploitation for privilege escalation': 'T1068',
    'security software discovery': 'T1518.001',
    'data from local system': 'T1533',
    'remote desktop protocol': 'T1021.001',
    'data compressed': 'T1560',  # Maps to T1560 - Archive Collected Data
    'software packing': 'T1027.002',
    'ping': 'S0097',
    'brute force': 'T1110',
    'commonly used port': 'T1571',
    'modify registry': 'T1112',
    'uncommonly used port': 'T1571',
    'process injection': 'T1055',
    'timestomp': 'T1070.006',
    'windows management instrumentation': 'T1047',
    'data staged': 'T1074',
    'rundll32': 'T1218.011',
    'regsvr32': 'T1218.010',
    'account discovery': 'T1087',
    'process discovery': 'T1057',
    'clipboard data': 'T1115',
    'binary padding': 'T1027.001',
    'pass the hash': 'T1550.002',
    'network service scanning': 'T1046',
    'system service discovery': 'T1007',
    'data encrypted': 'T1486',
    'system network connections discovery': 'T1049',
    'windows admin shares': 'T1021.002',
    'system owner/user discovery': 'T1033',
    'launch agent': 'T1543.001',
    'permission groups discovery': 'T1069',
    'indicator removal on host': 'T1070',
    'input capture': 'T1056',
    'virtualization/sandbox evasion': 'T1497.001',
    'dll side-loading': 'T1574.002',
    'scheduled task': 'T1053',
    'access token manipulation': 'T1134',
    'powershell': 'T1546.013',
    'exfiltration over alternative protocol': 'T1048',
    'hidden files and directories': 'T1564.001',
    'network share discovery': 'T1135',
    'query registry': 'T1012',
    'credentials in files': 'T1552.001',
    'audio capture': 'T1123',
    'video capture': 'T1125',
    'peripheral device discovery': 'T1120',
    'spearphishing via service': 'T1566.003',
    'data encrypted for impact': 'T1486',
    'data destruction': 'T1485',
    'template injection': 'T1221',
    'inhibit system recovery': 'T1490',
    'create account': 'T1136',
    'exploitation of remote services': 'T1210',
    'valid accounts': 'T1078',
    'dynamic data exchange': 'T1559.002',
    'office application startup': 'T1137',
    'data obfuscation': 'T1001',
    'domain trust discovery': 'T1482',
    'email collection': 'T1114',
    'man in the browser': 'T1185',
    'data from removable media': 'T1025',
    'bootkit': 'T1542.003',
    'logon scripts': 'T1037',
    'execution through module load': 'T1129',
    'llmnr/nbt-ns poisoning and relay': 'T1557.001',
    'external remote services': 'T1133',
    'domain fronting': 'T1090.004',
    'sid-history injection': 'T1134.005',
    'service stop': 'T1489',
    'disk structure wipe': 'T1561.002',
    'credentials in registry': 'T1552.002',
    'appinit dlls': 'T1546.010',
    'exploit public-facing application': 'T1190',
    'remote access tools': 'T1219',
    'signed binary proxy execution': 'T1218',
    'appcert dlls': 'T1546.009',
    'winlogon helper dll': 'T1547.004',
    'file permissions modification': 'T1222',
    'hooking': 'T1056.004',
    'system firmware': 'T1542.001',
    'lsass driver': 'T1547.008',
    'distributed component object model': 'T1021.003',
    'cmstp': 'T1218.003',
    'execution guardrails': 'T1480',
    'component object model hijacking': 'T1546.015',
    'accessibility features': 'T1546.008',  # TODO: Help wanted
    'keychain': 'T1555.001',
    'mshta': 'T1218.005',
    'pass the ticket': 'T1550.003',
    'kerberoasting': 'T1558.003',
    'password policy discovery': 'T1201',
    'local job scheduling': 'T1053.001',
    'windows remote management': 'T1021.006',
    'bits jobs': 'T1197',
    'data from information repositories': 'T1213',
    'lc_load_dylib addition': 'T1546.006',
    'histcontrol': 'T1562.003',
    'file system logical offsets': 'T1006',
    'regsvcs/regasm': 'T1218.009',
    'exploitation for credential access': 'T1212',
    'sudo': 'T1548.003',
    'installutil': 'T1218.004',
    'query registry ': 'T1012',
    'launchctl': 'T1569.001',
    '.bash_profile and .bashrc': 'T1546.004',
    'applescript': 'T1059.002',
    'emond': 'T1546.014',
    'control panel items': 'T1218.002',
    'application shimming': 'T1546.011',
}


class TrainingData(object):
    def __init__(self):
        self.mappings = {}  # Mapping is sentence text plus a list of Attack IDs

    def add_mapping(self, sentence_text, attack_id=None):
        mappings = self.mappings.get(sentence_text, [])  # Get mappings or empty list

        if attack_id:  # If attack_id is specified, add it to the list
            if attack_id not in mappings:
                mappings.append(attack_id)

        self.mappings[sentence_text] = mappings  # Put the mapping list back in


def get_attack_id(description):
    """Given a description, get the ATTACK ID. Raises IndexError if the retrieval fails."""
    lower_description = description.lower()
    attack_id = attack_lookup[lower_description]
    return attack_id


def main():
    with open('data/training/archive/all_analyzed_reports.json') as f:
        all_analyzed_reports = json.load(f)

    with open('data/training/archive/negative_data.json') as f:
        negative_data = json.load(f)

    training_data = TrainingData()

    # Add the positives
    for key, value in all_analyzed_reports.items():
        if key.endswith('-multi'):  # It's a multi-mapping, value is a dictionary
            technique_names = value['technique_names']
            sentences = value['sentances']  # Sentences is misspelled in the source data
            for sentence in sentences:
                for technique_name in technique_names:
                    technique_id = get_attack_id(technique_name)
                    training_data.add_mapping(sentence, technique_id)
        else:  # It's a single-mapping, value is a list of sentences
            technique_id = get_attack_id(key)
            for sentence in value:
                training_data.add_mapping(sentence, technique_id)

    # Add the negatives
    for sentence in negative_data:
        training_data.add_mapping(sentence, None)

    res = ReportExportSerializer(training_data)
    res.is_valid()

    with open(outfile, 'w') as f:
        json.dump(res.data, f, indent=4)


if __name__ == "__main__":
    main()
