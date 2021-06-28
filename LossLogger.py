from gensim.models.callbacks import CallbackAny2Vec

# gensim으로 word2vec 훈련 중 epoch마다 loss를 확인할 수 있는 callback
class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.loss_previous_step = 0

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        #print(f'Loss after epoch {self.epoch}: {loss - self.loss_previous_step}')
        print(f'Loss after epoch {self.epoch}: current loss : {loss}, previous loss : {self.loss_previous_step}, diff : {loss - self.loss_previous_step} ')
        self.epoch += 1
        self.loss_previous_step = loss        