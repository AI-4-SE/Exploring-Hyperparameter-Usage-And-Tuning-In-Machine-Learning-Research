import subprocess
import os
import glob
import subprocess
import sys
from git import Repo

# The folder where we store our results.
EVALUATION_FOLDER = "out"

TEST_REPOS = [
    'https://github.com/lucidrains/g-mlp-pytorch', 'https://github.com/albertpumarola/D-NeRF', 'https://github.com/rhythmcao/semantic-parsing-dual', 'https://github.com/bapanes/Gamma-Ray-Point-Source-Detector', 'https://github.com/juglab/DenoiSeg', 'https://github.com/dvlab-research/parametric-contrastive-learning', 'https://github.com/mansimane/WormML', 'https://github.com/gfxdisp/asap', 'https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation', 'https://github.com/Ugenteraan/Deep-CapsNet', 'https://github.com/DengPingFan/Polyp-PVT', 'https://github.com/yourh/AttentionXML', 'https://github.com/yuleiniu/rva', 'https://github.com/henrylee2570/NoisyAnchor', 'https://github.com/perfgao/lua-ffi-lightGBM', 'https://github.com/browatbn2/VLight', 'https://github.com/Vegeta2020/CIA-SSD', 'https://github.com/NVIDIA/tacotron2', 'https://github.com/DENG-MIT/Stiff-PINN', 'https://github.com/xavysp/DexiNed', 'https://github.com/blue-yonder/tsfresh', 'https://github.com/SCAN-NRAD/BrainRegressorCNN', 'https://github.com/BDBC-KG-NLP/TEN_EMNLP2020', 'https://github.com/WingsBrokenAngel/Semantics-AssistedVideoCaptioning', 'https://github.com/ychfan/scn', 'https://github.com/Julian-Theis/AVATAR', 'https://github.com/IbarakikenYukishi/differential-mdl-change-statistics', 'https://github.com/roggirg/count-ception_mbm', 'https://github.com/idobronstein/my_WRN', 'https://github.com/AndrewAtanov/simclr-pytorch', 'https://github.com/bhneo/Capsnet-Experiments', 'https://github.com/RicherMans/GPV', 'https://github.com/box-key/Subjective-Class-Issue', 'https://github.com/magic282/NeuSum', 'https://github.com/machine-reasoning-ufrgs/graph-neural-networks', 'https://github.com/Yongbinkang/ExpFinder', 'https://github.com/yongheng1991/qec_net', 'https://github.com/vinhdv1628/etnlp', 'https://github.com/TanUkkii007/wavenet', 'https://github.com/ferchonavarro/shape_aware_segmentation', 'https://github.com/BigRedT/no_frills_hoi_det', 'https://github.com/xinw1012/cycle-confusion', 'https://github.com/sharajpanwar/CC-WGAN-GP', 'https://github.com/therebellll/NegIoU-PosIoU-Miou', 'https://github.com/llyx97/Rosita', 'https://github.com/DavidJanz/molecule_grammar_rnn', 'https://github.com/facebookresearch/habitat-challenge', 'https://github.com/jramapuram/variational_saccading', 
    'https://github.com/LynnHo/PA-GAN-Tensorflow', 'https://github.com/baoyujing/hdmi', 'https://github.com/smt-HS/CE3', 'https://github.com/EmreTaha/shortcut-perspective-TF2', 'https://github.com/MoxinC/DST-SC', 'https://github.com/rguo12/network-deconfounder-wsdm20', 'https://github.com/voxmenthe/ncsn_1', 'https://github.com/EKirschbaum/LeMoNADe', 'https://github.com/ucl-bug/helmnet', 'https://github.com/anorthman/custom', 'https://github.com/sbuschjaeger/SubmodularStreamingMaximization', 'https://github.com/jxhuang0508/FSDR', 'https://github.com/zongdai/EditingForDNN', 'https://github.com/johnthebrave/nlidb-datasets', 'https://github.com/matwilso/relation-networks', 'https://github.com/q275212/rlcard', 'https://github.com/MyChocer/KGTN', 'https://github.com/AlexeyZhuravlev/visual-backprop', 'https://github.com/ctuning/ck-env', 'https://github.com/phueb/BabyBertSRL', 'https://github.com/valeoai/SemanticPalette', 'https://github.com/anandharaju/Basic_TCN', 'https://github.com/tracy6955/IBM_seizure_data', 'https://github.com/Network-Maritime-Complexity/Structural-core', 'https://github.com/saeedkhaki92/collaborative-filtering-for-yield-prediction', 'https://github.com/lca4/interank', 'https://github.com/mbsariyildiz/key-protected-classification', 'https://github.com/vlgiitr/ntm-pytorch', 'https://github.com/ahmedbhna/VGG_Paper', 'https://github.com/Kylin9511/ACRNet', 'https://github.com/Rediminds/All-Data-Inclusive-Deep-Learning-Models-to-Predict-Critical-Events-in-MIMIC-III', 'https://github.com/thu-media/FedCL', 'https://github.com/allenai/scruples', 'https://github.com/Smilels/multimodal-translation-teleop', 'https://github.com/mayoor/attention_network_experiments', 'https://github.com/arjunmajum/vln-bert', 'https://github.com/Kaixiong-Zhou/DGN', 'https://github.com/woodfrog/floor-sp', 'https://github.com/sophiaalexander/GANDissect-for-High-Schoolers', 'https://github.com/cheon-research/J-NCF-pytorch', 'https://github.com/Junyoungpark/CGS', 'https://github.com/guoqingbao/Multiception', 'https://github.com/bionlproc/BERT-CRel-Embeddings', 'https://github.com/thunlp/Neural-Snowball', 'https://github.com/maremun/quffka', 'https://github.com/odegeasslbc/Sketch2art-pytorch', 'https://github.com/jinze1994/ATRank', 'https://github.com/caoquanjie/ADDA-master', 'https://github.com/Komal7209/BackUp-Twitter-Sentiment-Analysis-', 'https://github.com/pvgladkov/density-peaks-sentence-clustering',
    'https://github.com/mdipcit/GTCNN', 'https://github.com/cyang-cityu/MetaCorrection', 'https://github.com/stevenkleinegesse/seqbed', 'https://github.com/zihangdai/xlnet', 'https://github.com/fmthoker/skeleton-contrast', 'https://github.com/eagle705/bert', 'https://github.com/Fraunhofer-AISEC/regression_data_poisoning', 'https://github.com/cersar/3D_detection', 'https://github.com/RoyJames/Fast3DScattering-release', 'https://github.com/jasjeetIM/recovering_compressible_signals', 'https://github.com/nar-k/persistent-excitation', 'https://github.com/frankaging/BERT_LRP', 'https://github.com/Chang-Chia-Chi/SaintPlus-Knowledge-Tracing-Pytorch', 'https://github.com/zh3nis/pat-sum', 'https://github.com/pse-ecn/pose-sensitive-embedding', 'https://github.com/mtli/sAP', 'https://github.com/MLI-lab/ConvDecoder', 'https://github.com/studian/CVND_P2_Image_Captioning', 'https://github.com/zekunhao1995/DualSDF', 'https://github.com/alexanderrichard/action-sets', 'https://github.com/NVlabs/dex-ycb-toolkit', 'https://github.com/davidhalladay/MMDetection', 'https://github.com/SamSamhuns/covid_19_hate_speech', 'https://github.com/Baichenjia/DCGAN-eager', 'https://github.com/Anjok07/ultimatevocalremovergui', 'https://github.com/JingzhaoZhang/why-clipping-accelerates', 'https://github.com/tattaka/Antialiased-CNNs-Converter-PyTorch', 'https://github.com/braincreators/octconv', 'https://github.com/PRIS-CV/PCA-Net', 'https://github.com/RuiLiFeng/LAE', 'https://github.com/johnarevalo/gmu-mmimdb', 'https://github.com/INK-USC/nl-explanation', 'https://github.com/rs9000/Neural-Turing-machine', 'https://github.com/GuoLiuFang/seglink-lfs', 'https://github.com/MAC-AutoML/YOCO-BERT', 'https://github.com/muyiguangda/pytorch-yolov3', 'https://github.com/alexis-jacq/Pytorch-Tutorials', 'https://github.com/siplab-gt/generative-causal-explanations', 'https://github.com/mongeoroo/Relevance-CAM', 'https://github.com/Shritesh99/100DaysofMLCodeChallenge', 'https://github.com/TonghanWang/ROMA', 'https://github.com/lucidrains/tab-transformer-pytorch', 'https://github.com/birlrobotics/PMN', 'https://github.com/ayulockin/Explore-NFNet', 'https://github.com/WANG-KX/SIREN-2D', 'https://github.com/Morpheus3000/intrinseg', 'https://github.com/dshelukh/CapsuleLearner', 'https://github.com/AlexKuhnle/film', 'https://github.com/juanmc2005/streamingspeakerdiarization', 'https://github.com/qiuzhen8484/COVID-DA', 
    'https://github.com/snap-stanford/GIB', 'https://github.com/vijaykeswani/Fair-Max-Entropy-Distributions', 'https://github.com/setharram/facenet', 'https://github.com/lsqshr/AH-Net', 'https://github.com/Teoroo-CMC/PiNN', 'https://github.com/saugatkandel/second-order-phase-retrieval', 'https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch', 'https://github.com/daniilidis-group/polar-transformer-networks', 'https://github.com/ari-dasci/s-rafni', 'https://github.com/iduta/pyconv', 'https://github.com/lettucecfd/lettuce', 'https://github.com/hrbigelow/ae-wavenet', 'https://github.com/freelunchtheorem/Conditional_Density_Estimation', 'https://github.com/SonyCSLParis/DrumGAN', 'https://github.com/overlappredator/OverlapPredator', 'https://github.com/SarielMa/ICLR2020_AI4AH', 'https://github.com/saizhang0218/TMC', 'https://github.com/doublebite/Sequence-to-General-tree', 'https://github.com/saeedkhaki92/CNN-RNN-Yield-Prediction', 'https://github.com/davrempe/contact-human-dynamics', 'https://github.com/Marie0909/AtrialJSQnet', 'https://github.com/tingyuansen/The_Payne', 'https://github.com/xiaobaicxy/SMN_Multi_Turn_Response_Selection_Pytorch', 'https://github.com/makgyver/rectorch', 'https://github.com/liuxinkai94/Graph-embedding', 'https://github.com/ericjang/draw', 'https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs', 'https://github.com/bigchem/transformer-cnn', 'https://github.com/ewsheng/nlg-bias', 'https://github.com/nutintin/Robotic_SP2019', 'https://github.com/ipc-lab/deepJSCC-feedback', 'https://github.com/AlfredXiangWu/LightCNN', 'https://github.com/pfnet-research/label-efficient-brain-tumor-segmentation', 'https://github.com/creme-ml/creme', 'https://github.com/lijiazheng99/Counterfactuals-for-Sentiment-Analysis', 'https://github.com/OIdiotLin/DeepLab-pytorch', 'https://github.com/NiallJeffrey/DeepMass', 'https://github.com/lanzhang128/disentanglement', 'https://github.com/facebookresearch/reconsider', 'https://github.com/benedekrozemberczki/PDN', 'https://github.com/airalcorn2/baller2vec', 'https://github.com/microsoft/FLAML', 'https://github.com/ShotDownDiane/tcn-master', 'https://github.com/YeongHyeon/Super-Resolution_CNN-PyTorch', 'https://github.com/vkristoll/cloud-masking-ANNs', 'https://github.com/facebookresearch/active-mri-acquisition', 'https://github.com/shiyemin/shuttlenet', 'https://github.com/birdortyedi/fashion-image-inpainting',
    'https://github.com/Ehsan-Yaghoubi/MAN-PAR-', 'https://github.com/wy-moonind/trackrcnn_with_deepsort', 'https://github.com/mloning/sktime-m4', 'https://github.com/dengyang17/OAAG', 'https://github.com/linjx-ustc1106/TuiGAN-PyTorch', 'https://github.com/hjc18/language_modeling_lstm', 'https://github.com/YivanZhang/lio', 'https://github.com/CharlieDinh/FEDL_pytorch', 'https://github.com/nikkkkhil/lane-detection-using-lanenet', 'https://github.com/usccolumbia/MOOCSP', 'https://github.com/satyadevntv/ROT', 'https://github.com/stanfordnlp/stanfordnlp', 'https://github.com/CuauSuarez/Bop2ndOrder', 'https://github.com/tansey/quantile-regression', 'https://github.com/marekrei/sequence-labeler', 'https://github.com/nickolor/Learning-to-See-in-the-Dark_and_DeepISP', 'https://github.com/kienduynguyen/Layer-Normalization', 'https://github.com/ai4ce/DeepMapping', 'https://github.com/KRLGroup/explainable-inference-on-sequential-data-via-memory-tracking', 'https://github.com/HoangTrinh/B_DRRN', 'https://github.com/varununleashed/tiny_faces_temp', 'https://github.com/tim-learn/BA3US', 'https://github.com/kushagra06/DDPG', 'https://github.com/kilgore92/Probabalistic-U-Net', 'https://github.com/cgomezsu/FIAQM', 'https://github.com/yunlu-chen/PointMixup', 'https://github.com/NeptuneProjects/RISClusterPT', 'https://github.com/kevinwong2013/COMS4995_Team_4_Zero_Shot_Classifier', 'https://github.com/aimagelab/STAGE_action_detection', 'https://github.com/williamcaicedo/ISeeU2', 'https://github.com/msultan/vde_metadynamics', 'https://github.com/thinng/GraphDTA', 'https://github.com/tingxueronghua/pytorch-classification-advprop', 'https://github.com/Bao-Jiarong/ResNet', 'https://github.com/martellab-sri/AMINN', 'https://github.com/lilaspourpre/kw_extraction', 'https://github.com/hukefei/taobaolive', 'https://github.com/KenzaB27/VAE-VampPrior', 'https://github.com/ShunChengWu/SCFusion_Network', 'https://github.com/zhengdao-chen/SRNN', 'https://github.com/inoryy/reaver', 'https://github.com/aswinapk/chatbot_rnn', 'https://github.com/hdcouture/TOCCA', 'https://github.com/yasumasaonoe/DenoiseET', 'https://github.com/Walid-Rahman2/modified_sentence_transfomers', 'https://github.com/biss/unet', 'https://github.com/Zartris/TD3_continuous_control', 'https://github.com/RiplleYang/DenseFusion', 'https://github.com/gvalvano/multiscale-adversarial-attention-gates', 'https://github.com/Xylon-Sean/rfnet', 
    'https://github.com/airsplay/VisualRelationships', 'https://github.com/ArchipLab-LinfengZhang/pytorch-scalable-neural-networks', 'https://github.com/team-approx-bayes/BayesBiNN', 'https://github.com/JCBrouwer/maua-stylegan2', 'https://github.com/manuelmolano/Spike-GAN', 'https://github.com/rahular/joint-coref-srl', 'https://github.com/Nico-Sch/RL-Chatbot', 'https://github.com/oclaudio/cubic-oscillator', 'https://github.com/Liberty3000/rl', 'https://github.com/xternalz/WideResNet-pytorch', 'https://github.com/He-jerry/DSSNet', 'https://github.com/jackd/ige', 'https://github.com/qq456cvb/AAE', 'https://github.com/universome/loss-patterns', 'https://github.com/jekunz/probing', 'https://github.com/Jacobew/AutoPanoptic', 'https://github.com/Music-and-Culture-Technology-Lab/omnizart', 'https://github.com/natalialmg/MMPF', 'https://github.com/parthchadha/upsideDownRL', 'https://github.com/yassouali/CCT', 'https://github.com/aboev/arae-tf', 'https://github.com/soumik12345/Enet-Tensorflow', 'https://github.com/dsshim0125/gaussian-ram', 'https://github.com/tfrerix/constrained-nets', 'https://github.com/zhenxingsh/Pytorch_DANet', 'https://github.com/JunweiLiang/Object_Detection_Tracking', 'https://github.com/juliaiwhite/amortized-rsa', 'https://github.com/mfinzi/constrained-hamiltonian-neural-networks', 'https://github.com/Zhuoranbupt/ECE285fa19', 'https://github.com/Sulam-Group/Adversarial-Robust-Supervised-Sparse-Coding', 'https://github.com/LONG-9621/VoteNet', 'https://github.com/LingxiaoShawn/PairNorm', 'https://github.com/tensorly/tensorly', 'https://github.com/HQXie0910/DeepSC', 'https://github.com/isp1tze/MAProj', 'https://github.com/Grayming/ALIL', 'https://github.com/balast/saliency_detector', 'https://github.com/vkristoll/cloud-masking-SOMs', 'https://github.com/shivram1987/diffGrad', 'https://github.com/SAP-samples/acl2019-commonsense-reasoning', 'https://github.com/TonythePlaneswalker/pcn', 'https://github.com/jfainberg/sincnet_adapt', 'https://github.com/david-klindt/NIPS2017', 'https://github.com/vanzytay/QuaternionTransformers', 'https://github.com/gulvarol/bsldict', 'https://github.com/liuyvchi/NROLL', 'https://github.com/andreamad8/Universal-Transformer-Pytorch', 'https://github.com/uuujf/IterAvg', 'https://github.com/bloomberg/cnn-rnf', 'https://github.com/KunZhou9646/seq2seq-EVC', 'https://github.com/hammerlab/fancyimpute', 'https://github.com/c-dickens/sketching_optimisation', 
    'https://github.com/allenai/cartography', 'https://github.com/mikefairbank/dlts_paper_code', 'https://github.com/uclanlp/synpg', 'https://github.com/noegroup/paper_boltzmann_generators', 'https://github.com/WeLoveKiraboshi/In-PlaneRotationAwareDepthEstimation', 'https://github.com/rpuiggari/bert2', 'https://github.com/cindyxinyiwang/multiview-subword-regularization', 'https://github.com/yangshunzhi1994/EdgeCNN', 'https://github.com/kevivk/mwp_adversarial', 'https://github.com/lollcat/RL-Process-Design', 'https://github.com/lzhengchun/dsgan', 'https://github.com/SJ001/AI-Feynman', 'https://github.com/threewisemonkeys-as/torched_impala', 'https://github.com/RaptorMai/online-continual-learning', 'https://github.com/cycraig/MP-DQN', 'https://github.com/LeoXuZH/CervicalCancerDetection', 'https://github.com/guillermo-navas-palencia/optbinning', 'https://github.com/iclavera/learning_to_adapt', 'https://github.com/hzxie/GRNet', 'https://github.com/auckland-cosmo/LearnAsYouGoEmulator', 'https://github.com/jkcrosby3/FashionMNST', 'https://github.com/liuch37/semantic-segmentation', 'https://github.com/Data-Science-in-Mechanical-Engineering/edge', 'https://github.com/ssrp/Multi-level-DCNet', 'https://github.com/jyhengcoder/lans_optimizer', 'https://github.com/Otsuzuki/Meta-learning-of-Pooling-Layers-for-Character-Recognition', 'https://github.com/lirus7/Heterogeneity-Loss-to-Handle-Intersubject-and-Intrasubject-Variability-in-Cancer', 'https://github.com/gurpreet-singh135/Image-Interpolation-via-adaptive-separable-convolution', 'https://github.com/stellargraph/stellargraph', 'https://github.com/CYBruce/STAWnet', 'https://github.com/rois-codh/kaokore', 'https://github.com/pratikkakkar/deep-diff', 'https://github.com/yf817/ICNet', 'https://github.com/ardyh/bert-ada', 'https://github.com/KaidiXu/GCN_ADV_Train', 'https://github.com/JieyuZ2/TMN', 'https://github.com/Jackn0/snn_optimal_conversion_pipeline', 'https://github.com/hukuda222/code2seq', 'https://github.com/RSMI-NE/RSMI-NE', 'https://github.com/lucidrains/learning-to-expire-pytorch', 'https://github.com/OceanParcels/parcels', 'https://github.com/revsic/jax-variational-diffwave', 'https://github.com/laohur/RelationNet', 'https://github.com/robintibor/braindecode', 'https://github.com/idiotprofessorchen/bert.github.io', 'https://github.com/renan-cunha/Bandits', 'https://github.com/BethanyL/damaged_cnns', 'https://github.com/Laborieux-Axel/Quantized_VGG', 
    'https://github.com/hinofafa/Self-Attention-HearthStone-GAN', 'https://github.com/mengfu188/mmdetection.bak', 'https://github.com/uzeful/IFCNN', 'https://github.com/kcmeehan/SmartDetect', 'https://github.com/nayeon7lee/bert-summarization', 'https://github.com/YUEXIUG/M2TS', 'https://github.com/taeoh-kim/temporal_data_augmentation', 'https://github.com/lance-ying/NHNN', 'https://github.com/jramapuram/BYOL', 'https://github.com/myedibleenso/this-before-that', 'https://github.com/zzs1994/D-LADMM', 'https://github.com/PuchatekwSzortach/voc_ssd', 'https://github.com/nccr-itmo/FEDOT', 'https://github.com/dingkeyan93/DISTS', 'https://github.com/odegeasslbc/OOGAN-pytorch', 'https://github.com/lisha-chen/Deep-structured-facial-landmark-detection', 'https://github.com/alinstein/Depth_estimation', 'https://github.com/Herge/GAN2', 'https://github.com/jspan/PHYSICS_SR', 'https://github.com/ms5898/ECBM6040-Project', 'https://github.com/basiralab/HUNet', 'https://github.com/DengPingFan/SINet', 'https://github.com/sjtuytc/segmentation-driven-pose', 'https://github.com/jasonwu0731/GettingToKnowYou', 'https://github.com/fuguoji/HSRL', 'https://github.com/umd-huang-lab/reinforcement-learning-via-spectral-methods', 'https://github.com/eric4421/MTWI-2018', 'https://github.com/aakashrkaku/knee-cartilage-segmentation', 'https://github.com/carinanorre/Brain-Tumour-Segmentation-Dissertation', 'https://github.com/MattAlexMiracle/SmartPatch', 'https://github.com/miritrope/genome', 'https://github.com/Hrach2003/change_detection', 'https://github.com/ryanhalabi/starcraft_super_resolution', 'https://github.com/AlexandraVolokhova/stochasticity_in_neural_ode', 'https://github.com/Lyusungwon/apex_dqn_pytorch', 'https://github.com/NVIDIA/waveglow', 'https://github.com/SmallMunich/nutonomy_pointpillars', 'https://github.com/ucl-exoplanets/pylightcurve-torch', 'https://github.com/mseg-dataset/mseg-api', 'https://github.com/FriedRonaldo/SinGAN', 'https://github.com/ozanciga/learning-to-segment', 'https://github.com/ShristiShrestha/SincConvBasedSpeakerRecognition', 'https://github.com/hshustc/CVPR19_Incremental_Learning', 'https://github.com/chridey/fever2-columbia', 'https://github.com/KhanJr/Generative-Adversarial-Networks-COMPUTER-VISION', 'https://github.com/rshaojimmy/AAAI2020-RFMetaFAS', 'https://github.com/Bakikii/stylegan2-pytorch23', 'https://github.com/kavitabala/geostyle', 'https://github.com/ks2labs/modules', 
    'https://github.com/sdi1100041/SLEIPNIR', 'https://github.com/yan-roo/SpineNet-Pytorch', 'https://github.com/rickyHong/pytorch-light-head-rcnn-repl', 'https://github.com/ConceptLengthLearner/ReproducibilityRepo', 'https://github.com/edricwu/Testing-1', 'https://github.com/dipjyoti92/StarGAN-Voice-Conversion', 'https://github.com/JieXuUESTC/DEMVC', 'https://github.com/chenziku/train-procgen', 'https://github.com/yanghr/SVD_Prune_EDLCV', 'https://github.com/Kennethborup/BYOL', 'https://github.com/Siomarry/Audio_recognition_', 'https://github.com/yashchandak/OptFuture_NSMDP', 'https://github.com/hendrycks/ethics', 'https://github.com/kasparmartens/c-GPLVM', 'https://github.com/parviagrawal/IdenProf', 'https://github.com/Tharun24/MACH', 'https://github.com/HoYoung1/image-embedding', 'https://github.com/otl-artorg/instrument-pose', 'https://github.com/YadiraF/PRNet', 'https://github.com/codertimo/ConveRT-pytorch', 'https://github.com/naver/dope', 'https://github.com/xkianteb/ApproPO', 'https://github.com/luo3300612/image-captioning-DLCT', 'https://github.com/aravind0706/flowpp', 'https://github.com/mcogswell/evolang', 'https://github.com/windwithforce/lane-detection', 'https://github.com/Gitgigabyte/mmd', 'https://github.com/quantlet/mlvsgarch', 'https://github.com/GilesStrong/calo_muon_regression', 'https://github.com/wdobbels/FIREnet', 'https://github.com/amrutn/Information-in-Language', 'https://github.com/nisheeth-golakiya/hybrid-sac', 'https://github.com/CVLAB-Unibo/omeganet', 'https://github.com/weiwenjiang/QML_tutorial', 'https://github.com/yanxp/PointNet', 'https://github.com/yue-zhongqi/ifsl', 'https://github.com/ElternalEnVy/tensorflow_rbm', 'https://github.com/demianbucik/collaborative-filtering-recommender-systems', 'https://github.com/MrTornado24/CS498_DL_Project', 'https://github.com/castorini/honk', 'https://github.com/yuyuta/moeadpy', 'https://github.com/sherrychen1120/MIDAS', 'https://github.com/mathiasunberath/DeepDRR', 'https://github.com/onue5/VideoQA', 'https://github.com/vinliao/object-detection-python', 'https://github.com/lucidrains/tab-transformer-pytorch/tree/main/tab_transformer_pytorch', 'https://github.com/ertug/Weak_Class_Source_Separation', 'https://github.com/CompressTeam/TransformCodingInference', 'https://github.com/facebookresearch/mtrl', 'https://github.com/junwenchen/GAP_SRAM', 'https://github.com/WangHelin1997/GL-AT', 'https://github.com/joshuajss/RTAA', 
    'https://github.com/evanmy/voxel_shape_analysis', 'https://github.com/CausalML/ESPRM', 'https://github.com/bestend/tf2-bi-lstm-crf-nni', 'https://github.com/xalanq/chinese-sentiment-classification', 'https://github.com/shrezaei/MI-on-EL', 'https://github.com/eladsegal/tag-based-multi-span-extraction', 'https://github.com/jaswindersingh2/RNAsnap2', 'https://github.com/WangZesen/SincNet-Tensorflow2'
]

def get_repo_name_from_url(url):
    """
    Analyze a repository with CfgNet.
    :param url: URL to the repository
    :return: Repository name
    """
    repo_name = url.split("/")[-1]
    repo_name = repo_name.split(".")[0]
    return repo_name


def process_repo(url):
    """
    Analyze a repository with CfgNet.
    :param url: URL to the repository
    :param commit: Hash of the lastest commit that should be analyzed
    :param ignorelist: List of file paths to ignore in the analysis
    """
    repo_name = get_repo_name_from_url(url)
    repo_folder = EVALUATION_FOLDER + "/" + repo_name
    results_folder = EVALUATION_FOLDER + "/results/" + repo_name
    abs_repo_path = os.path.abspath(repo_folder)

    # Cloning repository
    Repo.clone_from(url, repo_folder)

    # Init repository
    subprocess.run(
        f"cfgnet init -m {abs_repo_path}", shell=True, executable="/bin/bash"
    )

    # Copy results into result folder
    subprocess.run(["cp", "-r", repo_folder + "/.cfgnet", results_folder])

    # Remove repo folder
    remove_repo_folder(repo_folder)


def remove_repo_folder(repo_name):
    """Remove the cloned repository."""
    if os.path.exists(repo_name):
        subprocess.run(["rm", "-rf", repo_name])


def main():
    """Run the analysis."""
    # create evaluation folder
    if os.path.exists(EVALUATION_FOLDER):
        subprocess.run(["rm", "-rf", EVALUATION_FOLDER])
    subprocess.run(["mkdir", "-p", EVALUATION_FOLDER + "/results"])

    index = int(sys.argv[1])
    process_repo(TEST_REPOS[index])

if __name__ == "__main__":
    main()
