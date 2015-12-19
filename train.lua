require 'torch'
require 'image'
require 'pl' -- this is somehow responsible for lapp working in qlua mode
require 'paths'
ok, DISP = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
ADVERSARIAL = require 'adversarial'
DATASET = require 'dataset'
NN_UTILS = require 'utils.nn_utils'
MODELS = require 'models_ratio15'


----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  --save             (default "logs")       subdirectory to save logs
  --saveFreq         (default 30)           save every saveFreq epochs
  --network          (default "")           reload pretrained network
  --V_dir            (default "logs")       Directory where V networks are saved
  --G_pretrained_dir (default "logs")
  --noplot                                  plot while training
  --D_sgd_lr         (default 0.02)         D SGD learning rate
  --G_sgd_lr         (default 0.02)         G SGD learning rate
  --D_sgd_momentum   (default 0)            D SGD momentum
  --G_sgd_momentum   (default 0)            G SGD momentum
  --batchSize        (default 32)           batch size
  --N_epoch          (default 960)          Number of examples per epoch (-1 means all)
  --G_L1             (default 0)            L1 penalty on the weights of G
  --G_L2             (default 0e-6)         L2 penalty on the weights of G
  --D_L1             (default 0e-7)         L1 penalty on the weights of D
  --D_L2             (default 1e-4)         L2 penalty on the weights of D
  --D_iterations     (default 1)            number of iterations to optimize D for
  --G_iterations     (default 1)            number of iterations to optimize G for
  --D_maxAcc         (default 1.01)         Deactivate learning of D while above this threshold
  --D_clamp          (default 1)            Clamp threshold for D's gradient (+/- N)
  --G_clamp          (default 5)            Clamp threshold for G's gradient (+/- N)
  --D_optmethod      (default "adam")       adam|adagrad
  --G_optmethod      (default "adam")       adam|adagrad
  --threads          (default 4)            number of threads
  --gpu              (default 0)            gpu to run on (default cpu)
  --noiseDim         (default 100)          dimensionality of noise vector
  --window           (default 3)            window id of sample image
  --seed             (default 1)            seed for the RNG
  --colorSpace       (default "rgb")        rgb|yuv|hsl|y
  --nopretraining                           Whether to deactivate loading of pretrained networks
  --profile          (default "NONE")       snow32|snow64|trees32|trees64|baubles32|baubles64
]]

--[[
--height           (default 32)           height of images
--width            (default 48)           width of images
--]]

assert(OPT.profile == "snow32"
       or OPT.profile == "snow64"
       or OPT.profile == "trees32"
       or OPT.profile == "trees64"
       or OPT.profile == "baubles32"
       or OPT.profile == "baubles64",
       "--profile must be 'snow32', 'snow64', 'trees32', 'trees64', 'baubles32' or 'baubles64'")

if OPT.profile == "snow32" then
    OPT.height = 32
    OPT.width = 32+16
    OPT.dataDirs = {"dataset/preprocessed/snowy-landscapes"}
elseif OPT.profile == "snow64" then
    OPT.height = 64
    OPT.width = 64+32
    OPT.dataDirs = {"dataset/preprocessed/snowy-landscapes"}
elseif OPT.profile == "trees32" then
    OPT.height = 32
    OPT.width = 32
    OPT.dataDirs = {"dataset/preprocessed/christmas-trees"}
elseif OPT.profile == "trees64" then
    OPT.height = 64
    OPT.width = 64
    OPT.dataDirs = {"dataset/preprocessed/christmas-trees"}
elseif OPT.profile == "baubles32" then
    OPT.height = 32
    OPT.width = 32
    OPT.dataDirs = {"dataset/preprocessed/baubles"}
elseif OPT.profile == "baubles64" then
    OPT.height = 64
    OPT.width = 64
    OPT.dataDirs = {"dataset/preprocessed/baubles"}
else
    error("Unknown profile name")
end


NORMALIZE = false
if OPT.colorSpace == "y" then
    OPT.grayscale = true
end
START_TIME = os.time()

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

-- fix seed
math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)

-- threads
torch.setnumthreads(OPT.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- possible output of disciminator
CLASSES = {"0", "1"}
Y_GENERATOR = 0
Y_NOT_GENERATOR = 1

-- axis of images: 3 channels, <scale> height, <scale> width
if OPT.grayscale then
    IMG_DIMENSIONS = {1, OPT.height, OPT.width}
else
    IMG_DIMENSIONS = {3, OPT.height, OPT.width}
end
-- size in values/pixels per input image (channels*height*width)
INPUT_SZ = IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3]

----------------------------------------------------------------------
-- get/create dataset
----------------------------------------------------------------------
DATASET.colorSpace = OPT.colorSpace
DATASET.setFileExtension("jpg")
DATASET.setHeight(IMG_DIMENSIONS[2])
DATASET.setWidth(IMG_DIMENSIONS[3])
DATASET.setDirs(OPT.dataDirs)
----------------------------------------------------------------------

-- run on gpu if chosen
-- We have to load all kinds of libraries here, otherwise we risk crashes when loading
-- saved networks afterwards
print("<trainer> starting gpu support...")
require 'nn'
require 'cutorch'
require 'cunn'
require 'dpnn'
if OPT.gpu then
    cutorch.setDevice(OPT.gpu + 1)
    cutorch.manualSeed(OPT.seed)
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
end
torch.setdefaulttensortype('torch.FloatTensor')

function main()
    ----------------------------------------------------------------------
    -- Load / Define network
    ----------------------------------------------------------------------

    -- load previous networks (D and G)
    -- or initialize them new
    if OPT.network ~= "" then
        print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
        local tmp = torch.load(OPT.network)
        MODEL_D = tmp.D
        MODEL_G = tmp.G
        EPOCH = tmp.epoch + 1
        VIS_NOISE_INPUTS = tmp.vis_noise_inputs
        if NORMALIZE then
            NORMALIZE_MEAN = tmp.normalize_mean
            NORMALIZE_STD = tmp.normalize_std
        end

        if OPT.gpu == false then
            MODEL_D:float()
            MODEL_G:float()
        end
    else
        local pt_filename = paths.concat(OPT.save, string.format('pretrained_%dx%dx%d_nd%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim))
        -- pretrained via pretrain_with_previous_net.lua ?
        if not OPT.nopretraining and paths.filep(pt_filename) then
            local tmp = torch.load(pt_filename)
            MODEL_D = tmp.D
            MODEL_G = tmp.G
            MODEL_D:training()
            MODEL_G:training()
            if OPT.gpu == false then
                MODEL_D:float()
                MODEL_G:float()
            end
        else
            --------------
            -- D
            --------------
            MODEL_D = MODELS.create_D(OPT.profile, IMG_DIMENSIONS, OPT.gpu ~= false)

            --------------
            -- G
            --------------
            local g_pt_filename = paths.concat(OPT.G_pretrained_dir, string.format('g_pretrained_%dx%dx%d_nd%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim))
            if not OPT.nopretraining and paths.filep(g_pt_filename) then
                -- Load a pretrained version of G
                print("<trainer> loading pretrained G...")
                local tmp = torch.load(g_pt_filename)
                MODEL_G = tmp.G
                MODEL_G:training()
                if OPT.gpu == false then
                    MODEL_G:float()
                end
            else
                print("<trainer> Note: Did not find pretrained G")
                MODEL_G = MODELS.create_G(OPT.profile, IMG_DIMENSIONS, OPT.noiseDim, OPT.gpu ~= false)
            end
        end
    end

    print(MODEL_G)
    print(MODEL_D)

    -- count free parameters in D/G
    print(string.format('Number of free parameters in D: %d', NN_UTILS.getNumberOfParameters(MODEL_D)))
    print(string.format('Number of free parameters in G: %d', NN_UTILS.getNumberOfParameters(MODEL_G)))

    -- Copy models to GPU
    --[[
    if OPT.gpu then
        print("Copying model to gpu...")
        -- D is already on the GPU
        --MODEL_D = NN_UTILS.activateCuda(MODEL_D)
        MODEL_G = NN_UTILS.activateCuda(MODEL_G)
    end
    --]]

    -- loss function: negative log-likelihood
    CRITERION = nn.BCECriterion()

    -- retrieve parameters and gradients
    PARAMETERS_D, GRAD_PARAMETERS_D = MODEL_D:getParameters()
    PARAMETERS_G, GRAD_PARAMETERS_G = MODEL_G:getParameters()

    -- this matrix records the current confusion across classes
    CONFUSION = optim.ConfusionMatrix(CLASSES)

    -- Set optimizer states
    OPTSTATE = {
        adagrad = {
            D = { learningRate = 1e-3 },
            G = { learningRate = 1e-3 * 3 }
        },
        adadelta = { D = {}, G = {} },
        adamax = { D = {}, G = {} },
        adam = { D = {}, G = {} },
        rmsprop = {D = {}, G = {}},
        sgd = {
            D = {learningRate = OPT.D_sgd_lr, momentum = OPT.D_sgd_momentum},
            G = {learningRate = OPT.G_sgd_lr, momentum = OPT.G_sgd_momentum}
        }
    }

    if NORMALIZE then
        if NORMALIZE_MEAN == nil then
            TRAIN_DATA = DATASET.loadRandomImages(10000)
            NORMALIZE_MEAN, NORMALIZE_STD = TRAIN_DATA.normalize()
        end
    end

    if EPOCH == nil then
        EPOCH = 1
    end
    PLOT_DATA = {}
    if VIS_NOISE_INPUTS == nil then
        VIS_NOISE_INPUTS = NN_UTILS.createNoiseInputs(100)
    end

    -- training loop
    while true do
        print('Loading new training data...')
        TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch)
        if NORMALIZE then
            TRAIN_DATA.normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        end

        -- Show images and plots if requested
        if not OPT.noplot then
            NN_UTILS.visualizeProgress(VIS_NOISE_INPUTS)
        end

        -- Train D and G
        -- ... but train D only while having an accuracy below OPT.D_maxAcc
        --     over the last math.max(20, math.min(1000/OPT.batchSize, 250)) batches
        ADVERSARIAL.train(TRAIN_DATA, OPT.D_maxAcc, math.max(20, math.min(1000/OPT.batchSize, 250)))

        -- Save current net
        if EPOCH % OPT.saveFreq == 0 then
            local filename = paths.concat(OPT.save, 'adversarial.net')
            saveAs(filename)
        end

        EPOCH = EPOCH + 1
    end
end

-- Save the current models G and D to a file.
-- @param filename The path to the file
function saveAs(filename)
    os.execute(string.format("mkdir -p %s", sys.dirname(filename)))
    if paths.filep(filename) then
      os.execute(string.format("mv %s %s.old", filename, filename))
    end
    print(string.format("<trainer> saving network to %s", filename))
    NN_UTILS.prepareNetworkForSave(MODEL_G)
    NN_UTILS.prepareNetworkForSave(MODEL_D)
    torch.save(filename, {D = MODEL_D, G = MODEL_G, opt = OPT, plot_data = PLOT_DATA, epoch = EPOCH, vis_noise_inputs = VIS_NOISE_INPUTS, normalize_mean=NORMALIZE_MEAN, normalize_std=NORMALIZE_STD})
end

main()
