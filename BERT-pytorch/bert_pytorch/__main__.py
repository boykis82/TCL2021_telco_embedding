import argparse
from random import sample

from torch.utils.data import DataLoader

from .model import ALBERT
from .trainer import BERTTrainer
from .dataset import ALBERTDataset, WordVocab


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--dataset", required=True, type=str, help="dataset for train bert(train+test)")
    parser.add_argument("-t", "--train_ratio", type=float, default=0.8, help="train ratio")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

    parser.add_argument("-es", "--embedding_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")
    parser.add_argument("--aug_count", type=int, default=5, help="augmentation count")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="masking probability")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="sample ratio")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--warmup_rate", type=float, default=-1, help="warmup rate(-1 : no warmup)")

    parser.add_argument("-r", "--restart", type=bool, default=False, help="restart from last checkpoint?")

    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Dataset", args.dataset)
    max_pred = round(args.seq_len * args.mask_prob)
    print(f'seq_len = {args.seq_len}, mask_prob = {args.mask_prob}, max_pred = {max_pred}, aug_cnt = {args.aug_count}, sample_ratio = {args.sample_ratio}')
    train_dataset, test_dataset = \
        ALBERTDataset.create_dataset(
            corpus_path=args.dataset, 
            vocab=vocab, 
            seq_len=args.seq_len, 
            max_pred=max_pred, 
            mask_prob=args.mask_prob, 
            augmentation_count=args.aug_count, 
            train_ratio=args.train_ratio,
            sample_ratio=args.sample_ratio)
    print(f'Dataset loading completed! train size = {len(train_dataset)}, test_size = {len(test_dataset)}')

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f'Building ALBERT model. embed_size = {args.embedding_size}, hidden_size={args.hidden}, n_layers={args.layers}, attn_heads={args.attn_heads}, dropout={args.dropout}')
    bert = ALBERT(len(vocab), embed_size=args.embedding_size, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, seq_len=args.seq_len, dropout=args.dropout)

    total_steps = round(len(train_dataset) / args.batch_size) * args.epochs
    print(f'Creating ALBERT Trainer. lr = {args.lr}, weight_decay = {args.adam_weight_decay}, warmup_rate={args.warmup_rate}, total_steps={total_steps}, epoch = {args.epochs}')
    trainer = BERTTrainer(bert, args.embedding_size, vocab, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, warmup_rate=args.warmup_rate, total_steps=total_steps,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    if args.restart:
        last_ckpt_file_name, last_epoch = get_last_ckpt_file_name(args)        
        print(f'Training Restart From {last_epoch} epochs!')
        trainer.load(last_ckpt_file_name)
    else:
        print("Training Start")
        last_epoch = -1

    for epoch in range(last_epoch+1, args.epochs):
        current_step = trainer.train(epoch)
        trainer.save(epoch, args.output_path)
        
        if test_data_loader is not None:
            trainer.test(epoch)

        if current_step >= args.total_steps:
            break            



def get_last_ckpt_file_name(args):
    from os import listdir
    from os.path import isfile, join

    output_path = '/'.join(args.output_path.split('/')[:-1])
    output_file_prefix = args.output_path.split('/')[-1] + '.ep'
    onlyfiles = [f for f in listdir(output_path) if isfile(join(output_path, f))]
    ndx = max([int(e[len(output_file_prefix):]) for e in onlyfiles])
    return output_path + '/' + output_file_prefix + str(ndx), ndx