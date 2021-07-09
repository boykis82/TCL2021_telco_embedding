import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .optim import optim4GPU

from ..model import ALBERT, ALBERTLM

import tqdm


from torch.utils.tensorboard import SummaryWriter

class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: ALBERT, embed_size:int, vocab,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-3, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_rate: float = 0.05, total_steps:float = 200000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        self.vocab = vocab
        # Initialize the BERT Language Model, with BERT model
        self.model = ALBERTLM(bert, embed_size, len(self.vocab)).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        #self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim = optim4GPU(lr, warmup_rate, total_steps, self.model)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.mlm_criterion = nn.CrossEntropyLoss(reduction='none')
        self.sop_criterion = nn.CrossEntropyLoss()

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.summary_writer = SummaryWriter()


    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        '''
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        '''                              

        sum_loss = 0.0
        total_correct_sop = 0
        total_element_sop = 0
        total_correct_mlm = 0
        total_element_mlm = 0

        #for i, data in data_iter:
        for i, data in enumerate(data_loader):
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            #    sent_order_ouuput = B * 2
            #    mask_lm_output = B * seq_len * vocab_size
            sent_order_output, mask_lm_output = self.model.forward(data["input_ids"], data["segment_ids"], data["masked_pos"])

            '''
            output = {"input_ids"     : input_ids,
                  "segment_ids"   : segment_ids,
                  "input_mask"    : input_mask,
                  "masked_ids"    : masked_ids,
                  "masked_pos"    : masked_pos,
                  "masked_weights": masked_weights,
                  "ordered"       : ordered_label}            
            '''                  

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            #    ordered = B
            #    sent_order_output = B * 2                         
            sop_loss = self.sop_criterion(sent_order_output, data["ordered"])

            # 2-2. NLLLoss of predicting masked token word
            #    bert_label = B * seq_len
            #    mask_lm_output = B * seq_len * vocab_size            
            mlm_loss = self.mlm_criterion(mask_lm_output.transpose(1, 2), data["masked_ids"])
            mlm_loss = (mlm_loss * data["masked_weights"].float()).mean()

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = sop_loss + mlm_loss      

            # 3. backward and optimization only in train
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            sum_loss += loss.item()

            # SOP accuracy
            correct_sop = sent_order_output.argmax(dim=-1).eq(data["ordered"]).sum().item()
            total_correct_sop += correct_sop
            total_element_sop += data["ordered"].nelement()

            sop_acc = total_correct_sop / total_element_sop * 100

            # MLM accuracy
            mask_lm_output = mask_lm_output.argmax(dim=-1)
            mask_lm_output_mask = (mask_lm_output * data["masked_weights"]).gt(0)
            mask_lm_output = mask_lm_output.masked_select(mask_lm_output_mask)

            masked_ids_mask = data["masked_ids"].gt(0)
            masked_ids = data["masked_ids"].masked_select(masked_ids_mask)

            correct_mlm = mask_lm_output.eq(masked_ids).sum().item()
            total_correct_mlm += correct_mlm
            total_element_mlm += masked_ids.nelement()            

            mlm_acc = total_correct_mlm / total_element_mlm * 100
            

            post_fix = {
                "epoch": epoch,
                "iter": "[%d/%d]" % (i, len(data_loader)),
                "avg_loss": sum_loss / (i + 1),
                "sop_acc": sop_acc,
                "mlm_acc": mlm_acc,
                "total_loss": loss.item(),
                "mlm_loss": mlm_loss.item(),
                "sop_loss": sop_loss.item()
            }
            # tensorboard logging
            global_step = epoch * len(data_loader) + i
            '''
            self.summary_writer.add_scalars('data/scalar_grooup', 
                                            {'total_loss': loss.item(),
                                             'mlm_loss': mlm_loss.item(),
                                             'sop_loss': sop_loss.item(),
                                             'sop accuracy': sop_acc
                                            }, global_step)                       
            '''                                            
            self.summary_writer.add_scalar('total_loss', loss.item(), global_step)       
            self.summary_writer.add_scalar('mlm_loss', mlm_loss.item(), global_step)       
            self.summary_writer.add_scalar('sop_loss', sop_loss.item(), global_step)       
            self.summary_writer.add_scalar('lr', self.optim.get_lr()[0], global_step)       
            self.summary_writer.add_scalar('sop_acc', sop_acc, global_step)       
            self.summary_writer.add_scalar('mlm_acc', mlm_acc, global_step)       

            if i % self.log_freq == 0:
                print(post_fix)
                self.summary_writer.flush()    

        print("EP%d_%s, avg_loss=" % (epoch, str_code), sum_loss / len(data_loader), \
            "total_sop_acc=", total_correct_sop * 100.0 / total_element_sop, \
            "total_mlm_acc=", total_correct_mlm * 100.0 / total_element_mlm)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Weights Saved on:" % epoch, output_path)

        output_path = file_path + "_weightsonly" + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)

        return output_path

    def load(self, ckpt_file_path):
        self.model = torch.load(ckpt_file_path)
        print(f'Model Loaded From {ckpt_file_path}')