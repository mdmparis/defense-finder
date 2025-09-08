import os
import colorlog

from macsypy.scripts import macsyfinder
from warnings import simplefilter

from pyhmmer.easel import SequenceFile, TextSequence, Alphabet


import pandas as pd
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

df_dir = os.path.dirname(os.path.abspath(__file__))


def run(protein_file_name, dbtype, workers, coverage,
        adf, adf_only,
        esmdf, esmdf_only,
        tmp_dir, models_dir, nocut_ga, loglevel, index_dir, models_main_ver,
        base_outfile):

    scripts = []

    if adf_only == False and esmdf_only == False:
        if models_main_ver >= 2:
            scripts.append(['--db-type', dbtype, '--sequence-db',protein_file_name, '--models', 'defense-finder-models/DefenseFinder', 'all',
                            '--out-dir', os.path.join(tmp_dir, 'DefenseFinder'), '--w', str(workers),
                            '--coverage-profile', str(coverage), '--exchangeable-weight', '1'])
        else:
            gen_args = ['--db-type', dbtype, '--sequence-db', protein_file_name, '--models', 'defense-finder-models/DefenseFinder_{i}', 'all',
                    '--out-dir', os.path.join(tmp_dir, 'DF_{i}'), '--w', str(workers),
                    '--coverage-profile', str(coverage), '--exchangeable-weight', '1']
            scripts = [[f.format(i=i) for f in gen_args] for i in range(1, 6)]
        scripts.append(['--db-type', dbtype, '--sequence-db',protein_file_name, '--models', 'defense-finder-models/RM', 'all',
                        '--out-dir', os.path.join(tmp_dir, 'RM'), '--w', str(workers),
                        '--coverage-profile', str(coverage), '--exchangeable-weight', '1'])

        scripts.append(['--db-type', dbtype, '--sequence-db', protein_file_name, '--models', 'CasFinder', 'all',
                        '--out-dir', os.path.join(tmp_dir, 'Cas'), '-w', str(workers)])

    
    if (adf == True) or (adf_only == True):
        scripts.append(['--db-type', dbtype, '--sequence-db', protein_file_name, '--models', 'defense-finder-models/ADF', 'all',
                     '--out-dir', os.path.join(tmp_dir, 'AntiDefenseFinder'), '-w', str(workers)])

    for msf_cmd in scripts:
        if nocut_ga:
            msf_cmd.append("--no-cut-ga")
        if models_dir:
            msf_cmd.extend(("--models-dir", models_dir))
        if index_dir:
            if not os.path.exists(index_dir):
                os.makedirs(index_dir)
            msf_cmd.extend(("--index-dir", index_dir))
        if loglevel != "DEBUG":
            msf_cmd.append("--mute")

        macsyfinder.main(args=msf_cmd)

        # to avoid that the macsyfinder log messages
        # appear 7 times (one by msf call
        logger2 = colorlog.getLogger('macsypy')

        for h in logger2.handlers[:]:
            logger2.removeHandler(h)

    if (esmdf == True) or (esmdf_only == True):

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .ESM_DF.model import EsmForSequenceClassificationLightning

        logger = colorlog.getLogger("Defense_Finder")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ESM_35M
        checkpoint_path = os.path.join(df_dir, "ESM_DF", "weights_35M", "epoch=0-val_macro_ap=0.184.ckpt")

        # ESM_650Mdevice = torch.device("cuda" if use_cuda else "cpu")
        #checkpoint_path = "/pasteur/zeus/projets/p01/MDM/Projects/GeneCLR/scripts/finetuning_output/esm_finetuning_experiment_v7_high_octane/checkpoints/epoch=0-val_macro_ap=0.653.ckpt"

        # Load the trained model from checkpoint
        model = EsmForSequenceClassificationLightning.load_from_checkpoint(
            checkpoint_path, 
            strict=False
        )

        # Set to evaluation mode
        _ = model.eval()
        #model.bfloat16()

        tokenizer_path = os.path.join(df_dir, "ESM_DF", "tokenizer", "ESM2_tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        df_res = pd.DataFrame(columns=["protID", "notDef", "Def"])

        with SequenceFile(protein_file_name) as sf:
            seq = TextSequence()
            allseq = []
            allseqname = []
            current_batch = []
            current_batch_name = []
            i = 1
            nbatch = 0
            while sf.readinto(seq):
                # if i % 2: # batch size=2
                #     allseq.append([])
                #     allseqname.append([])
                sseq = seq.sequence
                sname = seq.name.decode()
                # allseq[-1].append(sseq)
                # allseqname[-1].append(sname)
                current_batch.append(sseq)
                current_batch_name.append(sname)
                seq.clear()

                if not i % 10: # batch size
                    nbatch += 1
                    logger.info(f"Predicting on batch {nbatch}. {i} proteins predicted so far")
                    batch = tokenizer(current_batch, padding=True, return_tensors="pt")
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    logger.info(f"Batch dimension : {input_ids.shape}")
                    outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
                    #logits = outputs.logits.float().detach().cpu().numpy() # if model in bfloat16
                    logits = outputs.logits.detach().cpu().numpy()
                    #sm_def = torch.softmax(logits, 1).T[1]
                    tmp_df = pd.concat([pd.Series(current_batch_name, name="protID"),
                                        pd.DataFrame(logits, columns=["notDef", "Def"])], axis=1)
                    df_res = pd.concat([df_res, tmp_df])
                    logger.info(f"df_res dimension : {df_res.shape}")
                    #df_res.set_index("protID").to_csv("res_esm.tsv", sep="\t", mode="a", header=False)
                    # reinit batch
                    current_batch = []
                    current_batch_name = []
                i += 1

         # Use 4/5 categories of likeliness
        logit_probable = 1 # to adjust after final training
        df_res["probable_def_gene"] = df_res.Def > logit_probable
        df_res[["protID", "Def", "probable_def_gene"]].to_csv(f"{base_outfile}_ESMDF.tsv", sep="\t", index=False)
