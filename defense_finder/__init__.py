import os
import colorlog

from macsypy.scripts import macsyfinder
from warnings import simplefilter, catch_warnings

from pyhmmer.easel import SequenceFile, TextSequence, Alphabet
from tqdm import tqdm
import pandas as pd
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=FutureWarning)
import sys


df_dir = os.path.dirname(os.path.abspath(__file__))


def seq_parser_inference_esm(protein_file_name, model, model_name, device, logger, tokenizer, batch_size=10):
    """
    parse protein fasta file, and run model in inference on it.
    """

    import torch

    df_res = pd.DataFrame(columns=["hit_id", "logit_NonDef", "logit_Def"])

    with SequenceFile(protein_file_name) as sf:
        seq = TextSequence()
        allseq = []
        allseqname = []
        current_batch = []
        current_batch_name = []
        i = 1
        nbatch = 0
        device_info = f"{torch.get_num_threads()} cpus" if device.type == "cpu" else torch.cuda.get_device_name()
        logger.info(f"Predicting with {model_name} on {protein_file_name}, running on {device_info}")
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
            
            if not i % batch_size: # batch size
                nbatch += 1
                logger.debug(f"Predicting on batch {nbatch}. {i} proteins predicted so far")
                batch = tokenizer(current_batch, padding=True, return_tensors="pt")
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                logger.debug(f"Batch dimension : {input_ids.shape}")
                outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
                #logits = outputs.logits.float().detach().cpu().numpy() # if model in bfloat16
                logits = outputs.logits.detach().cpu().numpy()
                #sm_def = torch.softmax(logits, 1).T[1]
                tmp_df = pd.concat([pd.Series(current_batch_name, name="hit_id"),
                                    pd.DataFrame(logits, columns=["logit_NonDef", "logit_Def"])], axis=1)
                df_res = pd.concat([df_res, tmp_df])
                logger.debug(f"df_res dimension : {df_res.shape}")
                #df_res.set_index("protID").to_csv("res_esm.tsv", sep="\t", mode="a", header=False)
                # reinit batch
                current_batch = []
                current_batch_name = []
            i += 1
        # predict on last partial batch
        if len(current_batch):
            logger.debug(f"Predicting on last batch {nbatch}. {i} proteins predicted so far")
            batch = tokenizer(current_batch, padding=True, return_tensors="pt")
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            logger.debug(f"Batch dimension : {input_ids.shape}")
            outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            #logits = outputs.logits.float().detach().cpu().numpy() # if model in bfloat16
            logits = outputs.logits.detach().cpu().numpy()
            #sm_def = torch.softmax(logits, 1).T[1]
            tmp_df = pd.concat([pd.Series(current_batch_name, name="hit_id"),
                                pd.DataFrame(logits, columns=["logit_NonDef", "logit_Def"])], axis=1)
            df_res = pd.concat([df_res, tmp_df])
            logger.debug(f"df_res dimension : {df_res.shape}")

    return df_res


def run_esm(protein_file_name, esm_model, loglevel, base_outfile):
        """
        run esm model on protein fasta file.
        """
        import torch
        from transformers import AutoTokenizer
        from .ESM_DF.model import EsmForSequenceClassificationLightning
        from huggingface_hub import hf_hub_download

        logger3 = colorlog.getLogger("EsmForSequenceClassificationLightning")
        logger3.setLevel("ERROR")
        logger = colorlog.getLogger("Defense_Finder")
        with catch_warnings(action="ignore"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        thresh_fdr = pd.read_json(os.path.join(df_dir, "config.json")).to_dict()
        # ESM_35M
        if esm_model == "35M":
            model_name = "ESM-DF_35M"
            checkpoint_path = hf_hub_download("jeanrjc/ESM-DF", filename="weights_35M/epoch=0-val_macro_ap=0.184.ckpt")
        elif esm_model == "650M":
            model_name = "ESM-DF_650M"
            checkpoint_path = hf_hub_download("jeanrjc/ESM-DF", filename="weights_650M/epoch=0-val_macro_ap=0.601.ckpt")
        else:
            logger.error(f"ESM_model should be 35M or 650M, not {esm_model}")
            sys.exit(1)

        with catch_warnings(action="ignore"):
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

        df_res = seq_parser_inference_esm(protein_file_name, model, model_name, 
                                      device, logger,
                                      tokenizer, batch_size=10)

        logger.info(f"ESM-DF prediction finished. {len(df_res)} proteins predicted")

        df_res["probable_defense_gene_F1"] = df_res.logit_Def >= thresh_fdr["F1"]["ESMDF"][esm_model]
        df_res["probable_defense_gene_FDR_1p"] = df_res.logit_Def >= thresh_fdr["FDR_99"]["ESMDF"][esm_model]
        df_res["probable_defense_gene_FDR_0.1p"] = df_res.logit_Def >= thresh_fdr["FDR_999"]["ESMDF"][esm_model]
        df_res.to_csv(f"{base_outfile}_ESMDF.tsv", sep="\t", index=False, float_format='%.5f')


def run_geneCLR(genes_df, loglevel, base_outfile):

        import torch
        from .GeneCLR_DF import geneclr_helper as gch
        from .GeneCLR_DF.geneclr.model import GeneCLR, GeneClrForTokenClassification
        from .GeneCLR_DF.geneclr.datamodule_inference import InferenceGeneCLRDataModule
        from .GeneCLR_DF.geneclr.prodigal_io import pyrodigal_annotation_dict_to_dataframe
        from huggingface_hub import hf_hub_download

        logger = colorlog.getLogger("Defense_Finder")
        with catch_warnings(action="ignore"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        thresh_fdr = pd.read_json(os.path.join(df_dir, "config.json")).to_dict()

        checkpoint_path = hf_hub_download("jeanrjc/GeneCLR-DF", filename="geneCLR_weights_inference.ckpt")

        WINDOW_SIZE = 64
        DEFAULT_STRIDE = 32

        windows, window_to_gene_indices = gch.create_overlapping_windows(
            genes_df, window_size=WINDOW_SIZE, stride=DEFAULT_STRIDE
        )
        logger.info(f"Created {len(windows)} windows (stride={DEFAULT_STRIDE})")
        gene_ids = genes_df["gene_id"].values
        
        datamodule = InferenceGeneCLRDataModule(
                            fragments=windows,
                            batch_size=10,
                            num_workers=0,
                            device=device,
                            fragment_length=WINDOW_SIZE,
                        )
        datamodule.setup()
        dataloader = datamodule.val_dataloader()
        df_res = gch.run_classifier_inference_cli(
            checkpoint_path,
            os.path.join(df_dir, "GeneCLR_DF", "finetuning_config_minimal.yaml"),
            device,
            dataloader,
            window_to_gene_indices,
            len(genes_df),
            gene_ids,
            f"{base_outfile}_GeneCLR_DF.tsv",
            "csv",
        )

        #logger.info(f"GeneCLR-DF prediction finished. {len(df_res)} proteins predicted")

        df_res["probable_defense_gene_F1"] = df_res.logit_Def >= thresh_fdr["F1"]["GENECLRDF"]
        df_res["probable_defense_gene_FDR_1p"] = df_res.logit_Def >= thresh_fdr["FDR_99"]["GENECLRDF"]
        df_res["probable_defense_gene_FDR_0.1p"] = df_res.logit_Def >= thresh_fdr["FDR_999"]["GENECLRDF"]
        df_res.to_csv(f"{base_outfile}_GeneCLR_DF.tsv", sep="\t", index=False, float_format='%.5f')




def run(protein_file_name, dbtype, workers, coverage,
        adf, adf_only,
        esmdf, esmdf_only, esm_model,
        geneclrdf, geneclrdf_only, genes_df,
        tmp_dir, models_dir, nocut_ga, loglevel, index_dir, models_main_ver,
        base_outfile):

    scripts = []

    if adf_only == False and esmdf_only == False and geneclrdf_only == False:
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

        run_esm(protein_file_name, esm_model, loglevel, base_outfile)
    
    if (geneclrdf == True) or (geneclrdf_only == True):
        run_geneCLR(genes_df, loglevel, base_outfile)