import json
import os
import pathlib
import re

import requests

# cve = "CVE-2022-0100"


# with open("/Users/ankitchauhan/Documents/va_parser/tsar.json", "r") as f:
#     data = json.load(f)

# for year in range(2022, 2000, -1):
#     year = str(year)

#     temp_dict = {}
#     cve_data = []

#     for cve_key in data:

#         reg_ex = re.search("CVE-" + year + "-", cve_key)

#         if reg_ex != None:
#             temp_dict[cve_key] = data[cve_key]
#             cve_data.append(temp_dict)
#             temp_dict = {}

#     with open(
#         "/Users/ankitchauhan/Drive/tram/va_parser/year-wise-cves/CVE_" + year + ".json",
#         "w",
#     ) as json_file:
#         json.dump(cve_data, json_file, indent=4)


def cveDataModification():

    path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "../va_parser/cve-data/"
    )

    files_list = os.listdir(path)

    for single_file in files_list:
        if ".json" in single_file:
            file_path = os.path.join(path, single_file)

            with open(file_path, "r") as f:
                cve_content = json.load(f)

            for i in range(len(cve_content["ttpMapping"])):
                cve_content["ttpMapping"][i]["confidence"] = cve_content["ttpMapping"][
                    i
                ]["confidence"]
                cve_content["ttpMapping"][i] = {
                    "techniqueId": cve_content["ttpMapping"][i]["technique_id"],
                    "confidence": cve_content["ttpMapping"][i]["confidence"],
                    "techniqueName": cve_content["ttpMapping"][i]["technique_name"],
                }

            with open(
                file_path,
                "w",
            ) as json_file:
                json.dump(cve_content, json_file, indent=4)
            print(file_path)


def get_empty_count():
    count = 0
    empty_list = []
    path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "../va_parser/cve-data/"
    )

    files_list = os.listdir(path)

    for single_file in files_list:
        if ".json" in single_file:
            file_path = os.path.join(path, single_file)

            with open(file_path, "r") as f:
                cve_content = json.load(f)

            if cve_content["ttpMapping"] == []:
                count += 1
                empty_list.append(single_file)
    print(empty_list)
    print("Total empty CVEs : ", count)


def analyse_unique_cve():
    count = 0
    empty_list = []
    path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "../va_parser/cve-data/"
    )
    with open("/Users/ankitchauhan/Drive/tram/va_parser/unique-cve-list.txt", "r") as f:
        unique_list = f.readlines()

    for single_file in unique_list:
        single_file = single_file.split("\n")[0] + ".json"
        print(single_file)
        file_path = os.path.join(path, single_file)

        try:

            with open(file_path, "r") as f:
                cve_content = json.load(f)
        except FileNotFoundError:
            print("File Not Found")
            continue

        if cve_content["ttpMapping"] == []:
            count += 1
            single_file = single_file.split(".json")[0] + "\n"
            empty_list.append(single_file)
    print(empty_list)
    with open(
        "/Users/ankitchauhan/Drive/tram/va_parser/not_have_ttp.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines(empty_list)

    print("Total empty CVEs : ", count)


def get_technique_name(technqiue_id):

    attack_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "../va_parser/attack-technique/"
    )
    file_path = os.path.join(attack_path, technqiue_id + ".json")

    technique_list = os.listdir(attack_path)

    if technqiue_id + ".json" in technique_list:

        with open(file_path, "r", encoding="utf-8-sig") as f:
            technique_data = json.load(f)

        return technique_data["name"]
    return ""


def cleanup():
    url = "http://localhost:8000/clear_memory/"

    payload = ""
    headers = {"Authorization": "Bearer " + get_access_token()}

    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.content)


def get_access_token():
    url = "http://localhost:8000/api/token/"

    payload = json.dumps({"username": "jojo", "password": "Safe@123"})
    headers = {"Cookie": "", "Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    response = json.loads(response.content)

    return response["access"]


def trade_web_missing_ttp():

    qwert_list = []
    temp = {}

    with open("/Users/ankitchauhan/Drive/tram/va_parser/not_have_ttp.txt", "r") as f:
        unique_list = f.readlines()

    with open("/Users/ankitchauhan/Documents/va_parser/tsar.json", "r") as f:
        tsar_json = json.load(f)

    url = "http://localhost:8000/techniques/"
    count = 0
    for single_cve in unique_list:
        cve_id = single_cve.split("\n")[0]
        if cve_id in tsar_json.keys():
            cve_desc = tsar_json[cve_id]["title"]

            payload = json.dumps(
                [
                    {
                        "unique_identifier": cve_id,
                        "description": [
                            {
                                "field_data": cve_desc,
                                "field_type": "control",
                            }
                        ],
                        "techniqueMapping": "",
                    }
                ]
            )
            headers = {
                "Authorization": "Bearer " + get_access_token(),
                "Content-Type": "application/json",
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            ttp_mapping = json.loads(response.content)[0]["technique_list"]

            final_json = tsar_json[cve_id]
            ttp_list = []

            for single_technique in ttp_mapping:
                temp_dict = single_technique
                technique_id = temp_dict["technique_id"]
                temp_dict["technique_name"] = get_technique_name(technique_id)
                ttp_list.append(temp_dict)

            final_json["ttpMapping"] = ttp_list

            out_path = os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "../va_parser/trade-web-missing/",
            )

            with open(
                out_path + cve_id + ".json",
                "w",
            ) as json_file:
                json.dump(final_json, json_file, indent=4)
            count += 1
            print(". . . . . ", count, " . . . . . ")
            print("Successfully wrote data for. . . ", cve_id)


trade_web_missing_ttp()
print("LAST . . . .. . . . ")
input()

year = 2002

for year in range(2017, 2020, 1):

    count = 0

    path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "../va_parser/year-wise-cves/"
    )

    filename = "CVE_" + str(year) + ".json"

    file_path = path + filename

    with open(file_path, "r") as f:
        cve_data = json.load(f)

    url = "http://localhost:8000/techniques/"

    for single_cve in cve_data:

        cve_path = os.path.join(
            pathlib.Path(__file__).parent.resolve(), "../va_parser/cve-data/"
        )

        exported_cve_list = os.listdir(cve_path)

        cve_id = list(single_cve.keys())[0]

        if cve_id + ".json" not in exported_cve_list:

            print("Writing data for. . . ", cve_id)

            final_json = {}
            cve_desc = single_cve[cve_id]["title"]

            payload = json.dumps(
                [
                    {
                        "unique_identifier": cve_id,
                        "description": [
                            {
                                "field_data": cve_desc,
                                "field_type": "control",
                            }
                        ],
                        "techniqueMapping": "",
                    }
                ]
            )
            headers = {
                "Authorization": "Bearer " + get_access_token(),
                "Content-Type": "application/json",
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            ttp_mapping = json.loads(response.content)[0]["technique_list"]

            final_json = single_cve[cve_id]
            ttp_list = []

            for single_technique in ttp_mapping:
                temp_dict = single_technique
                technique_id = temp_dict["technique_id"]
                temp_dict["technique_name"] = get_technique_name(technique_id)
                ttp_list.append(temp_dict)

            final_json["ttpMapping"] = ttp_list

            out_path = os.path.join(
                pathlib.Path(__file__).parent.resolve(), "../va_parser/cve-data/"
            )

            with open(
                out_path + cve_id + ".json",
                "w",
            ) as json_file:
                json.dump(final_json, json_file, indent=4)
            count += 1
            print(". . . . . ", count, " . . . . . ")
            print("Successfully wrote data for. . . ", cve_id)
