from .news_encoder import *

NEWS_ENCODER = {
    BERTje.code(): BERTje,
    RandomEmbedding.code(): RandomEmbedding,
    KimCNN.code(): KimCNN,
    NewsEncoderWuCNN.code(): NewsEncoderWuCNN
}
