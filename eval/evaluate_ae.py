import argparse
import time
import json
import numpy as np
import math
import random
import xml.etree.ElementTree as ET
from subprocess import check_output

from src.common import FILE_ENCODING


def label_rest_xml(fn, output_fn, corpus, label):
    dom = ET.parse(fn)
    root = dom.getroot()
    pred_y = []
    for zx, sent in enumerate(root.iter("sentence")):
        tokens = corpus[zx]
        lb = label[zx]
        opins = ET.Element("Opinions")
        token_idx, pt, tag_on = 0, 0, False
        start, end = -1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx < len(tokens) and pt >= len(tokens[token_idx]):
                pt = 0
                token_idx += 1

            if token_idx < len(tokens) and lb[token_idx] == 1 and pt == 0 and c != ' ':
                if tag_on:
                    end = ix
                    tag_on = False
                    opin = ET.Element("Opinion")
                    opin.attrib['target'] = sent.find('text').text[start:end]
                    opin.attrib['from'] = str(start)
                    opin.attrib['to'] = str(end)
                    opins.append(opin)
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and lb[token_idx] == 2 and pt == 0 and c != ' ' and not tag_on:
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and (lb[token_idx] == 0 or lb[token_idx] == 1) and tag_on and pt == 0:
                end = ix
                tag_on = False
                opin = ET.Element("Opinion")
                opin.attrib['target'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            elif token_idx >= len(tokens) and tag_on:
                end = ix
                tag_on = False
                opin = ET.Element("Opinion")
                opin.attrib['target'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            if c == ' ':
                pass
            elif tokens[token_idx][pt:pt + 2] == '``' or tokens[token_idx][pt:pt + 2] == "''":
                pt += 2
            else:
                pt += 1
        if tag_on:
            tag_on = False
            end = len(sent.find('text').text)
            opin = ET.Element("Opinion")
            opin.attrib['target'] = sent.find('text').text[start:end]
            opin.attrib['from'] = str(start)
            opin.attrib['to'] = str(end)
            opins.append(opin)
        sent.append(opins)
    dom.write(output_fn, encoding=FILE_ENCODING)


def label_laptop_xml(fn, output_fn, corpus, label):
    dom = ET.parse(fn)
    root = dom.getroot()
    pred_y = []
    for zx, sent in enumerate(root.iter("sentence")):
        tokens = corpus[zx]
        lb = label[zx]
        opins = ET.Element("aspectTerms")
        token_idx, pt, tag_on = 0, 0, False
        start, end = -1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx < len(tokens) and pt >= len(tokens[token_idx]):
                pt = 0
                token_idx += 1

            if token_idx < len(tokens) and lb[token_idx] == 1 and pt == 0 and c != ' ':
                if tag_on:
                    end = ix
                    tag_on = False
                    opin = ET.Element("aspectTerm")
                    opin.attrib['term'] = sent.find('text').text[start:end]
                    opin.attrib['from'] = str(start)
                    opin.attrib['to'] = str(end)
                    opins.append(opin)
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and lb[token_idx] == 2 and pt == 0 and c != ' ' and not tag_on:
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and (lb[token_idx] == 0 or lb[token_idx] == 1) and tag_on and pt == 0:
                end = ix
                tag_on = False
                opin = ET.Element("aspectTerm")
                opin.attrib['term'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            elif token_idx >= len(tokens) and tag_on:
                end = ix
                tag_on = False
                opin = ET.Element("aspectTerm")
                opin.attrib['term'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            if c == ' ' or ord(c) == 160:
                pass
            elif tokens[token_idx][pt:pt + 2] == '``' or tokens[token_idx][pt:pt + 2] == "''":
                pt += 2
            else:
                pt += 1
        if tag_on:
            tag_on = False
            end = len(sent.find('text').text)
            opin = ET.Element("aspectTerm")
            opin.attrib['term'] = sent.find('text').text[start:end]
            opin.attrib['from'] = str(start)
            opin.attrib['to'] = str(end)
            opins.append(opin)
        sent.append(opins)
    dom.write(output_fn, encoding=FILE_ENCODING)


def evaluate(pred_fn, command, template):
    with open(pred_fn, encoding=FILE_ENCODING) as f:
        pred_json = json.load(f)
    y_pred = []
    for ix, logit in enumerate(pred_json["logits"]):
        pred = [0] * len(pred_json["raw_X"][ix])
        for jx, idx in enumerate(pred_json["idx_map"][ix]):
            lb = np.argmax(logit[jx])
            if lb == 1:  # B
                pred[idx] = 1
            elif lb == 2:  # I
                if pred[idx] == 0:  # only when O->I (I->I and B->I ignored)
                    pred[idx] = 2
        y_pred.append(pred)

    # for ix, logit in enumerate(pred_json["logits"]):
    #     pred = [-1] * len(pred_json["raw_X"][ix])
    #     for jx, idx in enumerate(pred_json["idx_map"][ix]):
    #         lb = np.argmax(logit[jx])
    #         if pred[idx] == -1:
    #             pred[idx] = lb
    #     y_pred.append(pred)

    if 'REST' in command:
        command = command.split()
        label_rest_xml(template, command[6], pred_json["raw_X"], y_pred)
        acc = check_output(command).split()
        return float(acc[9][10:])
    elif 'Laptops' in command:
        command = command.split()
        label_laptop_xml(template, command[4], pred_json["raw_X"], y_pred)
        acc = check_output(command).split()
        return float(acc[15])


def evaluate2(pred_fn, command, template):
    with open(pred_fn, encoding=FILE_ENCODING) as f:
        pred_json = json.load(f)

    raw_X = pred_json["raw_X"]
    y_pred = pred_json["y_pred"]

    if 'REST' in command:
        command = command.split()
        label_rest_xml(template, command[6], raw_X, y_pred)
        acc = check_output(command).split()
        return float(acc[9][10:])
    elif 'Laptops' in command:
        command = command.split()
        label_laptop_xml(template, command[4], raw_X, y_pred)
        acc = check_output(command).split()
        return float(acc[15])


if __name__ == "__main__":
    # pred_fn = "ae/test/prediction.json"
    # command = "java -cp eval/eval.jar Main.Aspects ae/test/laptop_pred.xml ae/test/Laptops_Test_Gold.xml"
    # template = "ae/test/Laptops_Test_Data_PhaseA.xml"
    # evaluate2(pred_fn, command, template)

    # command = "java -cp eval/eval.jar Main.Aspects ae/official_data/laptop_pred.xml ae/official_data/Laptops_Test_Gold.xml"
    # command2 = "java -cp eval/eval.jar Main.Aspects ae/test/laptop_pred.xml ae/test/Laptops_Test_Gold.xml"
    # acc = check_output(command).split()
    # print(acc)
    # acc = check_output(command2).split()
    # print(acc)

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default="laptop")
    parser.add_argument('--run_dir', type=str, default="pt_ae")
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--no_context', default=False, action='store_true',
                        help="Whether to run validation.")

    args = parser.parse_args()
    domain = args.domain
    run_dir = args.run_dir
    if 'rest' == args.domain:
        command = "java -cp eval/A.jar absa16.Do Eval -prd ae/official_data/rest_pred.xml -gld ae/official_data/EN_REST_SB1_TEST.xml.gold -evs 2 -phs A -sbt SB1"
        template = "ae/official_data/EN_REST_SB1_TEST.xml.A"
    elif 'laptop' == args.domain:
        command = "java -cp eval/eval.jar Main.Aspects ae/official_data/laptop_pred.xml ae/official_data/Laptops_Test_Gold.xml"
        template = "ae/official_data/Laptops_Test_Data_PhaseA.xml"
    elif 'laptop_vn' == args.domain:
        command = "java -cp eval/eval.jar Main.Aspects ae/laptop_vn_official_data/laptop_pred.xml " \
                  "ae/laptop_vn_official_data/Laptops_Test_Gold.xml"
        template = "ae/laptop_vn_official_data/Laptops_Test_Gold.xml"

    dir = f"run_history\\{run_dir}\\{domain}\\9_runs_4_epoch_0_seq_start_100_window_no-context"
    #dir = f"run\\{run_dir}\\{domain}"

    if args.no_context:
        print("\n----------------Accuracy without sentence-in-context---------------")
        acc = 0
        # for i in range(args.runs):
        #     acc += evaluate2(f"{dir}\\{i + 1}\\predictions_NC.json", command,
        #                      template)
        # print(f"Accuracy in {args.runs} runs: {acc / args.runs}")

        for i in range(args.runs):
            acc = evaluate2(f"{dir}\\{i + 1}\\predictions_NC.json", command,
                             template)
            print(f"Accuracy in {i + 1} runs: {acc}")
    else:
        print("\n----------------Accuracy with sentence-in-context---------------")
        method_names = ['CMV', 'CMVP', 'F', 'FP', "start_position_0", "start_position_32", "start_position_64",
                        "start_position_96"]
        #method_names = ['start_position_32']
        for i, method_name in enumerate(method_names):
            acc = 0
            for i in range(args.runs):
                acc += evaluate2(f"{dir}\\{i + 1}\\predictions_{method_name}.json", command,
                                 template)
            print(f"Accuracy with {method_name} in {args.runs} runs: {acc / args.runs}")

            # for i in range(args.runs):
            #     acc = evaluate2(f"{dir}\\{i + 1}\\predictions_{method_name}.json", command,
            #                      template)
            #     print(f"Accuracy with {method_name} in runs {i + 1}: {acc}")
