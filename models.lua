require 'torch'
require 'nn'
require 'dpnn'
require 'cudnn'

local models = {}

function models.create_G_decoder_upsampling32x32(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 4x4
    model:add(nn.Linear(noiseDim, 256*4*4))
    model:add(nn.BatchNormalization(256*4*4))
    model:add(cudnn.ReLU(true))
    model:add(nn.View(256, 4, 4))

    -- 4x4 -> 8x8
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    -- 8x8 -> 16x16
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    -- 16x16 -> 32x32
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_decoder_upsampling32x32_branched(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    model:add(nn.Linear(noiseDim, 128))
    model:add(cudnn.ReLU(true))

    local conc = nn.Concat(2)
    for i=1,8 do
        local branch = nn.Sequential()

        -- 8x8
        branch:add(nn.Linear(128, 16*8*8))
        branch:add(nn.BatchNormalization(16*8*8))
        branch:add(cudnn.ReLU(true))
        branch:add(nn.View(16, 8, 8))

        -- 8x8 -> 16x16
        branch:add(nn.SpatialUpSamplingNearest(2))
        branch:add(cudnn.SpatialConvolution(16, 16, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
        branch:add(nn.SpatialBatchNormalization(16))
        branch:add(cudnn.ReLU(true))

        -- 16x16 -> 32x32
        branch:add(nn.SpatialUpSamplingNearest(2))
        branch:add(cudnn.SpatialConvolution(16, 16, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
        branch:add(nn.SpatialBatchNormalization(16))
        branch:add(cudnn.ReLU(true))

        conc:add(branch)
    end

    model:add(conc)

    model:add(cudnn.SpatialConvolution(16*8, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialConvolution(32, dimensions[1], 1, 1, 1, 1, 0, 0))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_decoder_upsampling32x48(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 4x4
    model:add(nn.Linear(noiseDim, 512*4*6))
    model:add(nn.BatchNormalization(512*4*6))
    model:add(nn.PReLU())
    --model:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --model:add(nn.SpatialBatchNormalization(512))
    --model:add(nn.PReLU())
    model:add(nn.View(512, 4, 6))

    -- 4x4 -> 8x8
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU())

    -- 8x8 -> 16x16
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU())

    -- 16x16 -> 32x32
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU())

    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_decoder_upsampling32x48_residual(dimensions, noiseDim, cuda)
    function createReduce(nbInputPlanes, nbOutputPlanes)
        local seq = nn.Sequential()
        seq:add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, 1, 1, 1, 1, 0, 0))
        seq:add(nn.SpatialBatchNormalization(nbOutputPlanes))
        seq:add(nn.PReLU())
        return seq
    end

    function createResidual(nbInputPlanes, nbInnerPlanes, nbOutputPlanes)
        local seq = nn.Sequential()
        local inner = nn.Sequential()
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInnerPlanes, 1, 1, 1, 1, 0, 0))
        inner:add(nn.SpatialBatchNormalization(nbInnerPlanes))
        inner:add(cudnn.ReLU(true))
        inner:add(cudnn.SpatialConvolution(nbInnerPlanes, nbInnerPlanes, 3, 3, 1, 1, 1, 1))
        inner:add(nn.SpatialBatchNormalization(nbInnerPlanes))
        inner:add(cudnn.ReLU(true))
        inner:add(cudnn.SpatialConvolution(nbInnerPlanes, nbOutputPlanes, 1, 1, 1, 1, 0, 0))
        inner:add(nn.SpatialBatchNormalization(nbOutputPlanes))
        inner:add(cudnn.ReLU(true))

        local conc = nn.ConcatTable(2)
        conc:add(inner)
        if nbInputPlanes == nbOutputPlanes then
            conc:add(nn.Identity())
        else
            reducer = nn.Sequential()
            reducer:add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, 1, 1, 1, 1, 0, 0))
            reducer:add(nn.SpatialBatchNormalization(nbOutputPlanes))
            reducer:add(cudnn.ReLU(true))
            conc:add(reducer)
        end
        seq:add(conc)
        seq:add(nn.CAddTable())
        return seq
    end

    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 4x6
    model:add(nn.Linear(noiseDim, 512*4*6))
    model:add(nn.BatchNormalization(512*4*6))
    model:add(cudnn.ReLU(true))
    model:add(nn.View(512, 4, 6))

    -- 4x6 -> 8x12
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(512))
    model:add(cudnn.ReLU(true))

    -- 8x12 -> 16x24
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    -- 16x24 -> 32x48
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(cudnn.ReLU(true))
    model:add(createResidual(128, 32, 128))
    model:add(createResidual(128, 32, 128))
    model:add(createResidual(128, 32, 128))
    model:add(createResidual(128, 32, 128))
    model:add(createResidual(128, 32, 128))

    -- decrease to usually 3 dimensions (image channels)
    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_decoder_upsampling64x64_residual(dimensions, noiseDim, cuda)
    function createResidual(nbInputPlanes, nbInnerPlanes, nbOutputPlanes)
        local seq = nn.Sequential()
        local inner = nn.Sequential()
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInnerPlanes, 1, 1, 1, 1, 0, 0))
        inner:add(nn.SpatialBatchNormalization(nbInnerPlanes))
        inner:add(cudnn.ReLU(true))
        inner:add(cudnn.SpatialConvolution(nbInnerPlanes, nbInnerPlanes, 3, 3, 1, 1, 1, 1))
        inner:add(nn.SpatialBatchNormalization(nbInnerPlanes))
        inner:add(cudnn.ReLU(true))
        inner:add(cudnn.SpatialConvolution(nbInnerPlanes, nbOutputPlanes, 1, 1, 1, 1, 0, 0))
        inner:add(nn.SpatialBatchNormalization(nbOutputPlanes))
        inner:add(cudnn.ReLU(true))

        local conc = nn.ConcatTable(2)
        conc:add(inner)
        if nbInputPlanes == nbOutputPlanes then
            conc:add(nn.Identity())
        else
            reducer = nn.Sequential()
            reducer:add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, 1, 1, 1, 1, 0, 0))
            reducer:add(nn.SpatialBatchNormalization(nbOutputPlanes))
            reducer:add(cudnn.ReLU(true))
            conc:add(reducer)
        end
        seq:add(conc)
        seq:add(nn.CAddTable())
        return seq
    end

    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 8x8
    model:add(nn.Linear(noiseDim, 512*8*8))
    model:add(nn.BatchNormalization(512*8*8))
    model:add(nn.PReLU())
    model:add(nn.View(512, 8, 8))

    -- 8x8 -> 16x16
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU())

    -- 16x16 -> 32x32
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU())

    -- 32x32 -> 64x64
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.PReLU())

    model:add(createResidual(64, 16, 64))
    model:add(createResidual(64, 16, 64))
    model:add(createResidual(64, 16, 64))
    model:add(createResidual(64, 16, 64))
    model:add(createResidual(64, 16, 64))

    -- decrease to usually 3 dimensions (image channels)
    model:add(cudnn.SpatialConvolution(64, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_decoder_upsampling64x64(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 8x8
    model:add(nn.Linear(noiseDim, 256*8*8))
    model:add(nn.BatchNormalization(256*8*8))
    --model:add(nn.PReLU())
    model:add(cudnn.ReLU(true))
    model:add(nn.View(256, 8, 8))

    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    --model:add(nn.PReLU())
    model:add(cudnn.ReLU(true))

    -- 8x8 -> 16x16
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    --model:add(nn.PReLU())
    model:add(cudnn.ReLU(true))

    -- 16x16 -> 32x32
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    --model:add(nn.PReLU())
    model:add(cudnn.ReLU(true))

    -- 32x32 -> 64x64
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(64))
    --model:add(nn.PReLU())
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialConvolution(64, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates the decoder part of an upsampling height-32px G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_decoder_upsampling64x96(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 4x6
    model:add(nn.Linear(noiseDim, 256*4*6))
    model:add(nn.BatchNormalization(256*4*6))
    model:add(cudnn.ReLU(true))
    model:add(nn.View(256, 4, 6))

    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    -- 4x6 -> 8x12
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    -- 8x12 -> 16x24
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    -- 16x24 -> 32x48
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(cudnn.ReLU(true))

    -- 32x48 -> 64x96
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialConvolution(64, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates G, which is identical to the decoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G(profile, dimensions, noiseDim, cuda)
    if profile == "snow32" then
        return models.create_G_decoder_upsampling32x48(dimensions, noiseDim, cuda)
    elseif profile == "snow64" then
        return models.create_G_decoder_upsampling64x96(dimensions, noiseDim, cuda)
    elseif profile == "trees32" then
        return models.create_G_decoder_upsampling32x32(dimensions, noiseDim, cuda)
    elseif profile == "trees64" then
        return models.create_G_decoder_upsampling64x64(dimensions, noiseDim, cuda)
    elseif profile == "baubles32" then
        return models.create_G_decoder_upsampling32x32_branched(dimensions, noiseDim, cuda)
    elseif profile == "baubles64" then
        return models.create_G_decoder_upsampling64x64(dimensions, noiseDim, cuda)
    else
        error("Unknown profile '" .. profile .. "'")
    end
end

-- Creates D.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_D(profile, dimensions, cuda)
    if profile == "snow32" then
        return models.create_D32x48(dimensions, cuda)
    elseif profile == "snow64" then
        return models.create_D64x96(dimensions, cuda)
    elseif profile == "trees32" then
        return models.create_D32x32(dimensions, cuda)
    elseif profile == "trees64" then
        return models.create_D64x64(dimensions, cuda)
    elseif profile == "baubles32" then
        return models.create_D32x32b(dimensions, cuda)
    elseif profile == "baubles64" then
        return models.create_D64x64(dimensions, cuda)
    else
        error("Unknown profile '" .. profile .. "'")
    end
end

function models.create_D32x32(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x48
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))

    -- 16x24
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 8x12
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.3))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 4x6
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.5))
    conv:add(nn.SpatialMaxPooling(2, 2))

    --[[
    conv:add(nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.5))
    conv:add(nn.SpatialMaxPooling(2, 2))
    --]]

    -- 2x3
    conv:add(nn.View(512*4*4))
    conv:add(nn.Linear(512*4*4, 256))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout(0.25))
    conv:add(nn.Linear(256, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32x32b(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    --conv:add(nn.Dropout(0.3))
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.ELU())
    conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.ELU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.Dropout(0.5))

    -- 16x16
    conv:add(nn.View(128*16*16))
    conv:add(nn.Linear(128*16*16, 256))
    conv:add(nn.ELU())
    conv:add(nn.Dropout(0.25))
    conv:add(nn.Linear(256, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32x48(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x48
    conv:add(nn.SpatialConvolution(dimensions[1], 256, 5, 5, 2, 2, (5-1)/2, (5-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.2))

    -- 16x24
    conv:add(nn.SpatialConvolution(256, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.2))
    --conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 8x12
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 4x6
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.2))
    --conv:add(nn.SpatialMaxPooling(2, 2))

    -- 2x3
    conv:add(nn.View(512*8*12))
    conv:add(nn.Linear(512*8*12, 64))
    conv:add(nn.PReLU())
    conv:add(nn.Linear(64, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D64x64(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 64x64
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 32x32
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 32x32
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.5))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    conv:add(nn.View(512*4*4))
    conv:add(nn.Linear(512*4*4, 64))
    conv:add(nn.PReLU())
    conv:add(nn.Linear(64, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D64x96(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 64x96
    --conv:add(nn.SpatialConvolution(dimensions[1], 256, 5, 5, 2, 2, (5-1)/2, (5-1)/2))
    --conv:add(nn.PReLU())
    --conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 32x48
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 16x24
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x12
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.50))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 4x6
    conv:add(nn.View(512*4*6))
    conv:add(nn.Linear(512*4*6, 64))
    conv:add(nn.PReLU())
    conv:add(nn.Linear(64, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

return models
