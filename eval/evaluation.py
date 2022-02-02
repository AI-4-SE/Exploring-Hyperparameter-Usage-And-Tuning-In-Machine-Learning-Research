import subprocess
import os
import glob
import subprocess
import sys
from git import Repo

# The folder where we store our results.
EVALUATION_FOLDER = "out"


TEST_REPOS = [
    "https://github.com/ibayer/PAKDD2015",
    "https://github.com/david-klindt/NIPS2017",
    "https://github.com/ahmedbhna/VGG_Paper",
    "https://github.com/hdcouture/TOCCA",
    "https://github.com/ewsheng/nlg-bias",
    "https://github.com/TranSys2020/TranSys",
    "https://github.com/xkianteb/ApproPO",
    "https://github.com/JieXuUESTC/DEMVC",
    "https://github.com/RuiLiFeng/LAE",
    "https://github.com/linjx-ustc1106/TuiGAN-PyTorch",
    "https://github.com/CausalML/ESPRM",
    "https://github.com/zhennany/synthetic",
    "https://github.com/CompressTeam/TransformCodingInference",
    "https://github.com/vijaykeswani/Fair-Max-Entropy-Distributions",
    "https://github.com/carinanorre/Brain-Tumour-Segmentation-Dissertation",
    "https://github.com/RaptorMai/online-continual-learning",
    "https://github.com/idea-iitd/GCOMB",
    "https://github.com/HQXie0910/DeepSC",
    "https://github.com/usccolumbia/MOOCSP",
    "https://github.com/ari-dasci/s-rafni",
    "https://github.com/jacobandreas/neuralese",
    "https://github.com/demianbucik/collaborative-filtering-recommender-systems",
    "https://github.com/satyadevntv/ROT",
    "https://github.com/SJ001/AI-Feynman",
    "https://github.com/dipjyoti92/StarGAN-Voice-Conversion",
    "https://github.com/xuehaolan/DANet",
    "https://github.com/YeongHyeon/Super-Resolution_CNN-PyTorch",
    "https://github.com/mloning/sktime-m4",
    "https://github.com/hendrycks/ethics",
    "https://github.com/box-key/Subjective-Class-Issue",
    "https://github.com/animeshprasad/citation_analysis",
    "https://github.com/yoandinkov/ranlp-2019",
    "https://github.com/AlexandraVolokhova/stochasticity_in_neural_ode",
    "https://github.com/Herge/GAN2",
    "https://github.com/lirus7/Heterogeneity-Loss-to-Handle-Intersubject-and-Intrasubject-Variability-in-Cancer",
    "https://github.com/Julian-Theis/AVATAR",
    "https://github.com/WangHelin1997/GL-AT",
    "https://github.com/sbuschjaeger/SubmodularStreamingMaximization",
    "https://github.com/hshustc/CVPR19_Incremental_Learning",
    "https://github.com/idobronstein/my_WRN",
    "https://github.com/BaoWangMath/LS-MCMC",
    "https://github.com/cgomezsu/FIAQM",
    "https://github.com/creme-ml/creme",
    "https://github.com/charliezhaoyinpeng/PDFM",
    "https://github.com/Ehsan-Yaghoubi/MAN-PAR-",
    "https://github.com/ShristiShrestha/SincConvBasedSpeakerRecognition",
    "https://github.com/Chang-Chia-Chi/SaintPlus-Knowledge-Tracing-Pytorch",
    "https://github.com/thehimalayanleo/Private-Generative-Models",
    "https://github.com/quantlet/mlvsgarch",
    "https://github.com/Shritesh99/100DaysofMLCodeChallenge",
    "https://github.com/mseitzer/csmri-refinement",
    "https://github.com/xalanq/chinese-sentiment-classification",
    "https://github.com/stellargraph/stellargraph",
    "https://github.com/paramitamirza/alignarr",
    "https://github.com/pratikkakkar/deep-diff",
    "https://github.com/thunlp/Neural-Snowball",
    "https://github.com/natalialmg/MMPF",
    "https://github.com/lzzcd001/ade-code",
    "https://github.com/renan-cunha/Bandits",
    "https://github.com/KunZhou9646/seq2seq-EVC",
    "https://github.com/Yongbinkang/ExpFinder",
    "https://github.com/fxia22/NeuralFDR",
    "https://github.com/ConceptLengthLearner/ReproducibilityRepo",
    "https://github.com/ardyh/bert-ada",
    "https://github.com/ms5898/ECBM6040-Project",
    "https://github.com/tomaszkolonko/DeepDIVA_asbestos",
    "https://github.com/birlrobotics/PMN",
    "https://github.com/jasonwu0731/GettingToKnowYou",
    "https://github.com/liuxinkai94/Graph-embedding",
    "https://github.com/ferchonavarro/shape_aware_segmentation",
    "https://github.com/pfnet-research/label-efficient-brain-tumor-segmentation",
    "https://github.com/allenai/cartography",
    "https://github.com/SCAN-NRAD/BrainRegressorCNN",
    "https://github.com/riron1206/kaggle_MoA",
    "https://github.com/fmthoker/skeleton-contrast",
    "https://github.com/aschein/bptf",
    "https://github.com/nccr-itmo/FEDOT",
    "https://github.com/manuelmolano/Spike-GAN",
    "https://github.com/gulvarol/bsldict",
    "https://github.com/llyx97/Rosita",
    "https://github.com/BigRedT/no_frills_hoi_det",
    "https://github.com/sgiguere/RobinHood-NeurIPS-2019",
    "https://github.com/kavitabala/geostyle",
    "https://github.com/lilaspourpre/kw_extraction",
    "https://github.com/stevenkleinegesse/seqbed",
    "https://github.com/Network-Maritime-Complexity/Structural-core",
    "https://github.com/setharram/facenet",
    "https://github.com/RicherMans/GPV",
    "https://github.com/epierson9/cyclic_HMMs",
    "https://github.com/NeptuneProjects/RISClusterPT",
    "https://github.com/vkristoll/cloud-masking-SOMs",
    "https://github.com/fhou80/EntEmb",
    "https://github.com/skearnes/color-features",
    "https://github.com/CharlieDinh/FEDL_pytorch",
    "https://github.com/JunweiLiang/Object_Detection_Tracking",
    "https://github.com/Siomarry/Audio_recognition_",
    "https://github.com/lightaime/sgas",
    "https://github.com/wdobbels/FIREnet",
    "https://github.com/JCBrouwer/maua-stylegan2",
    "https://github.com/mengfu188/mmdetection.bak",
    "https://github.com/jiongqian/MILE",
    "https://github.com/vinhdv1628/etnlp",
    "https://github.com/Near32/ReferentialGym",
    "https://github.com/frankaging/BERT_LRP",
    "https://github.com/lanzhang128/disentanglement",
    "https://github.com/guillermo-navas-palencia/optbinning",
    "https://github.com/perfgao/lua-ffi-lightGBM",
    "https://github.com/alibaba/x-deeplearning",
    "https://github.com/mayoor/attention_network_experiments",
    "https://github.com/fuguoji/HSRL",
    "https://github.com/Near32/ReferentialGym/tree/master/zoo/referential-games%2Bst-gs",
    "https://github.com/xyz-zy/distant-temprel",
    "https://github.com/juglab/DenoiSeg",
    "https://github.com/chridey/fever2-columbia",
    "https://github.com/furkanbiten/GoodNews",
    "https://github.com/bloomberg/cnn-rnf",
    "https://github.com/WingsBrokenAngel/Semantics-AssistedVideoCaptioning",
    "https://github.com/Data-Science-in-Mechanical-Engineering/edge",
    "https://github.com/Walid-Rahman2/modified_sentence_transfomers",
    "https://github.com/lachinov/brats2018-graphlabunn",
    "https://github.com/AMLab-Amsterdam/SEVDL_MGP",
    "https://github.com/rguo12/network-deconfounder-wsdm20",
    "https://github.com/yourh/AttentionXML",
    "https://github.com/jinze1994/ATRank",
    "https://github.com/bapanes/Gamma-Ray-Point-Source-Detector",
    "https://github.com/freelunchtheorem/Conditional_Density_Estimation",
    "https://github.com/allenai/scruples",
    "https://github.com/davrempe/contact-human-dynamics",
    "https://github.com/airalcorn2/baller2vec",
    "https://github.com/M-cube-wustl/ML_intermetallics",
    "https://github.com/OceanParcels/parcels",
    "https://github.com/Junyoungpark/CGS",
    "https://github.com/scenarios/PGNAS",
    "https://github.com/dvlab-research/parametric-contrastive-learning",
    "https://github.com/iyyun/Part-CNN",
    "https://github.com/c-dickens/sketching_optimisation",
    "https://github.com/baoyujing/hdmi",
    "https://github.com/Tharun24/MACH",
    "https://github.com/blue-yonder/tsfresh",
    "https://github.com/Gauffret/TrajectorySet",
    "https://github.com/tracy6955/IBM_seizure_data",
    "https://github.com/windwithforce/lane-detection",
    "https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs",
    "https://github.com/arjunmajum/vln-bert",
    "https://github.com/rahular/joint-coref-srl",
    "https://github.com/amrutn/Information-in-Language",
    "https://github.com/Ishan-Kumar2/Molecular_VAE_Pytorch",
    "https://github.com/yuyuta/moeadpy",
    "https://github.com/YivanZhang/lio",
    "https://github.com/shrezaei/MI-on-EL",
    "https://github.com/engrjavediqbal/MLSL",
    "https://github.com/kevinwong2013/COMS4995_Team_4_Zero_Shot_Classifier",
    "https://github.com/benedekrozemberczki/PDN",
    "https://github.com/jkcrosby3/FashionMNST",
    "https://github.com/overlappredator/OverlapPredator",
    "https://github.com/cindyxinyiwang/multiview-subword-regularization",
    "https://github.com/edricwu/Testing-1",
    "https://github.com/browatbn2/VLight",
    "https://github.com/ozanciga/learning-to-segment",
    "https://github.com/yue-zhongqi/ifsl",
    "https://github.com/HLTCHKUST/Mem2Seq",
    "https://github.com/albanyhep/JSDML",
    "https://github.com/SonyCSLParis/DrumGAN",
    "https://github.com/HKUST-KnowComp/Motif-based-PageRank",
    "https://github.com/valeoai/SemanticPalette",
    "https://github.com/sminer3/duplicate_statement_detection",
    "https://github.com/IbarakikenYukishi/differential-mdl-change-statistics",
    "https://github.com/classner/up",
    "https://github.com/nikkkkhil/lane-detection-using-lanenet",
    "https://github.com/smt-HS/CE3",
    "https://github.com/baokuiwang/context_aware_situation_entity",
    "https://github.com/martellab-sri/AMINN",
    "https://github.com/lance-ying/NHNN",
    "https://github.com/johnarevalo/gmu-mmimdb",
    "https://github.com/searchivarius/NonMetricSpaceLib",
    "https://github.com/choderalab/gimlet",
    "https://github.com/Sulam-Group/Adversarial-Robust-Supervised-Sparse-Coding",
    "https://github.com/snap-stanford/GIB",
    "https://github.com/gvalvano/multiscale-adversarial-attention-gates",
    "https://github.com/pvgladkov/density-peaks-sentence-clustering",
    "https://github.com/qiuzhen8484/COVID-DA",
    "https://github.com/aakashrkaku/knee-cartilage-segmentation",
    "https://github.com/Fraunhofer-AISEC/regression_data_poisoning",
    "https://github.com/sharajpanwar/CC-WGAN-GP",
    "https://github.com/gurpreet-singh135/Image-Interpolation-via-adaptive-separable-convolution",
    "https://github.com/GemsLab/StrucEmbedding-GraphLibrary",
    "https://github.com/vkristoll/cloud-masking-ANNs",
    "https://github.com/ShotDownDiane/tcn-master",
    "https://github.com/noegroup/paper_boltzmann_generators",
    "https://github.com/gauraviitg/BMTT-PETS-2017-surveillance-challenge",
    "https://github.com/maremun/quffka",
    "https://github.com/zihangdai/xlnet",
    "https://github.com/dengyang17/OAAG",
    "https://github.com/uuujf/IterAvg",
    "https://github.com/myedibleenso/this-before-that",
    "https://github.com/JairParra/GloVe-ML_Reproducibiliaty_Challenge",
    "https://github.com/alexwdong/IncubatorCVProject",
    "https://github.com/microsoft/FLAML",
    "https://github.com/mbsariyildiz/key-protected-classification",
    "https://github.com/MrTornado24/CS498_DL_Project",
    "https://github.com/wy-moonind/trackrcnn_with_deepsort",
    "https://github.com/aied2021TRMRC/AIED_2021_TRMRC_code",
    "https://github.com/reidpr/quac",
    "https://github.com/yiyang-gt/feat2vec",
    "https://github.com/AndrewAtanov/simclr-pytorch",
    "https://github.com/juliaiwhite/amortized-rsa",
    "https://github.com/basiralab/HUNet",
    "https://github.com/SamSamhuns/covid_19_hate_speech",
    "https://github.com/umd-huang-lab/reinforcement-learning-via-spectral-methods",
    "https://github.com/anandharaju/Basic_TCN",
    "https://github.com/SamanJamalAbbasi/DRESS_DeepRL",
    "https://github.com/hammerlab/fancyimpute",
    "https://github.com/ai4ce/DeepMapping",
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
        f"cfgnet init {abs_repo_path}", shell=True, executable="/bin/bash"
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
