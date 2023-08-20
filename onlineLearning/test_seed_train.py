from feedback import Framework, OurArguments
from tasks import get_task,Sample
import torch
import numpy as np
import random
import argparse
import tasks
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
from utils import *
from trainer import OurTrainer
import random

'''

Seed training class for initial training of mind2web to get the intial loss going

'''

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args

def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


def main():
    
    args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task_name)
    framework = Framework(args, task)
    
   #task = get_task("Mind2Web")
   
    #task.load_dataset()
    #train_sets = task.sample_train_sets(num_train=10, num_dev=10, num_eval=10, num_train_sets=1, seed=42)
    s1 = Sample(id=210, data={'context': '(html (body (div (div (div id=0 (div (span Plan Your ) (span Thrills at ) ) (div select (div Great Escape ) (div (a option (div (span Hurricane Harbor Phoenix ) (span Phoenix, AZ ) ) ) (div (a option (div (span Six Flags Magic Mountain ) (span Los Angeles, CA ) ) ) (a option (div (span Hurricane Harbor Los Angeles ) (span Los Angeles, CA ) ) ) (a option (div (span Six Flags Discovery Kingdom ) (span San Francisco/Sacramento, CA ) ) ) (a option (div (span Hurricane Harbor Concord ) (span San Francisco/Sacramento, CA ) ) ) ) (div (a option (div (span Six Flags Over Georgia ) (span Atlanta, GA ) ) ) (a option (div (span Six Flags White Water ) (span Atlanta, GA ) ) ) ) (div (a option (div (span Six Flags Great America ) (span Chicago, IL ) ) ) (a option (span Hurricane Harbor Chicago ) ) ) ) ) ) (span Park Hours ) ) (ul (li General Parking ) (li Unlimited visits to Six Flags Great Escape, Six Flags New England and LaRonde ) (li 15% Food & Merchandise discounts ) (li id=1 Unlimited visits for all of 2023 ) (li Unlimited visits to Hurricane Harbor ) (li 1 Skip the Line Pass ) (li 2 Specialty Rate Tickets ) ) (li id=2 (span Thrill Rides ) ) ) (div (div id=3 button close store ) (iframe accesso e-commerce store overlay (div id=4 (div (div navigation (div button (span menu ) ) (button Skip to main content ) (div menubar (div menuitem daily tickets (span Daily Tickets ) ) (div menuitem season passes (span Season Passes ) ) (div menuitem group tickets (span Group Tickets ) ) (div menuitem all season dining (span All Season Dining ) ) (promo button (span Promo ) ) ) (div (span button (span help ) ) (span button 0 cart items cart (span cart ) ) ) ) (div (div navigation (form form promoform (input text enter promotional code ) (button submit promo and close ) ) (a Privacy Statement ) ) (div main (ng-form theform ) (button button Back ) ) ) ) (div contentinfo (div (button button High Contrast Mode ) (a Privacy Statement ) ) ) ) ) ) ) )', 'question': "Based on the HTML webpage above, try to complete the following task:\nTask: Buy a diamond pass in New York's, Great escape park, add one meal dining plan to it, and select the flexible payment plan for Jame Jones. The email address is jame_jones@hotmail.com, zip code 10005 and age is 35.\nPrevious actions:\n[span]  Great Escape -> CLICK\n[button]  Go! -> CLICK\n[link]  Tickets & Passes \uf078 -> CLICK\n[link]  Tickets & Passes -> CLICK\n[span]  Purchase Pass -> CLICK\nWhat should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):\n\nA. None of the above\nB. (div id=0 (div (span Plan Your ) (span Thrills at\nC. (li id=1 Unlimited visits for all of 2023 )\nD. (li id=2 (span Thrill Rides ) )\nE. (div id=3 button close store )\nF. (div id=4 (div (div navigation (div button (span menu )\n", 'answer': 'A. None'}, correct_candidate='A. None', candidates=['B.\nAction: CLICK', 'B.\nAction: TYPE', 'B.\nAction: SELECT', 'C.\nAction: CLICK', 'C.\nAction: TYPE', 'C.\nAction: SELECT', 'D.\nAction: CLICK', 'D.\nAction: TYPE', 'D.\nAction: SELECT', 'A. None'])
    
    
    s2 = Sample(id=453, data={'context': '(html (app-home (main main (div (ul id=0 (li id=1 (a (span Vacation Deals ) (span Vacations ) ) ) (a Shop Hotels ) (a (span Rent a Car ) (span Cars ) ) (a GIFT CARDS ) ) (div id=2 (div (span TRAVEL WITH CONFIDENCE WITH THE FLY DELTA APP ) (div Download the app to find your gate, track your bags, upgrade your seats and more. ) ) ) ) ) (div id=3 (div (div (div (h3 About Delta ) (a About Delta ) ) (ul (a About Us ) (a Careers ) (a News Hub ) (a Investor Relations ) (a Business Travel ) (a Travel Agents ) (a Mobile App ) ) (ul (a About Us ) (a Careers ) (a News Hub ) (a Investor Relations ) (a Business Travel ) (a Travel Agents ) (a Mobile App ) ) ) (div (h3 Customer Service ) (a Customer Service ) ) (a id=4 Site Map ) ) ) ) )', 'question': "Based on the HTML webpage above, try to complete the following task:\nTask: find my trip with confirmation number SFTBAO including first and last name Joe Lukeman\nPrevious actions:\n[tab]  MY TRIPS -> CLICK\n[combobox]  Find Your Trip By -> CLICK\n[option]  Confirmation Number -> CLICK\n[input]   -> TYPE: SFTBAO\nWhat should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):\n\nA. None of the above\nB. (ul id=0 (li id=1 (a (span Vacation Deals ) (span\nC. (li id=1 (a (span Vacation Deals ) (span Vacations )\nD. (div id=2 (div (span TRAVEL WITH CONFIDENCE WITH THE FLY\nE. (div id=3 (div (div (div (h3 About Delta ) (a\nF. (a id=4 Site Map )\n", 'answer': 'A. None'}, correct_candidate='A. None', candidates=['B.\nAction: CLICK', 'B.\nAction: TYPE', 'B.\nAction: SELECT', 'C.\nAction: CLICK', 'C.\nAction: TYPE', 'C.\nAction: SELECT', 'D.\nAction: CLICK', 'D.\nAction: TYPE', 'D.\nAction: SELECT', 'A. None'])
    s3 = Sample(id=294, data={'context': '(html (body (div (table grid (tr row (a id=0 (span Wicked ) (span Fabulous Fox Theatre - St. Louis ) (span St. Louis, MO ) ) ) ) (iframe fb:page facebook social plugin f22569b367eaa4 (div feed (div (div id=1 Ticket Center ) (div id=2 ticketcenter.com ) ) (div id=3 (span (a perfectly unbalanced jeff dunham at (div Show Dates and Times: Friday, June 16th 2017 at 7:30pm Friday, June 23rd, 2017 at 7:30pm Friday, June 30th, 2017 ) (img may be an image of ) ) (div (div (div ticketcenter.com ) (div (a Perfectly Unbalanced Jeff Dunham at The Colosseum ) (div Show Dates and Times: Friday, June 16th 2017 at 7:30pm Friday, June 23rd, 2017 at 7:30pm Friday, June 30th, 2017 ) ) ) (a perfectly unbalanced jeff dunham at (div Show Dates and Times: Friday, June 16th 2017 at 7:30pm Friday, June 23rd, 2017 at 7:30pm Friday, June 30th, 2017 ) ) ) ) ) ) ) ) (tr (td 9 ) (td 10 ) (td 11 ) (td id=4 12 ) (td 13 ) (td 14 ) ) ) )', 'question': "Based on the HTML webpage above, try to complete the following task:\nTask: Browse the venues that are playing the Wicked show from Oct 5 to Oct 24 2023\nPrevious actions:\n[textbox]  Select Date Range -> CLICK\n[columnheader]  \ue080 -> CLICK\n[columnheader]  \ue080 -> CLICK\n[columnheader]  \ue080 -> CLICK\n[gridcell]  1 -> CLICK\nWhat should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):\n\nA. None of the above\nB. (a id=0 (span Wicked ) (span Fabulous Fox Theatre -\nC. (div id=1 Ticket Center )\nD. (div id=2 ticketcenter.com )\nE. (div id=3 (span (a perfectly unbalanced jeff dunham at (div\nF. (td id=4 12 )\n", 'answer': 'A. None'}, correct_candidate='A. None', candidates=['B.\nAction: CLICK', 'B.\nAction: TYPE', 'B.\nAction: SELECT', 'C.\nAction: CLICK', 'C.\nAction: TYPE', 'C.\nAction: SELECT', 'D.\nAction: CLICK', 'D.\nAction: TYPE', 'D.\nAction: SELECT', 'A. None'])
    s4 = Sample()
        #print(train_sets)
    train_sets = [[s1,s2]]
    print('Got train sets (test_seed_train)')
    
    eval_samples = [s3]
    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            '''
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples
            '''
            if args.trainer != "none":
                if args.num_dev is not None:
                    # Dev samples
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None

                # Training
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)
                
                print('Finishing Training')
                if not args.no_eval:
                    metrics = framework.evaluate(train_samples, eval_samples) 
                    
                    # No in-context learning if there is training
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples) 
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)
    
if __name__ == "__main__": 
    main()