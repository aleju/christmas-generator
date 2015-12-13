require 'torch'
require 'nn'
require 'dpnn'
require 'cudnn'

local models = {}

-- Creates the encoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_encoder16(dimensions, noiseDim)
    local model = nn.Sequential()
    local activation = nn.LeakyReLU

    model:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(activation())

    model:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))

    model:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))

    model:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(activation())

    model:add(nn.View(64 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    model:add(nn.Linear(64 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 512))
    model:add(nn.BatchNormalization(512))
    model:add(activation())
    model:add(nn.Linear(512, noiseDim))
    --model:add(nn.Dropout(0.2))

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates the encoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_encoder32(dimensions, noiseDim)
    local model = nn.Sequential()
    local activation = nn.LeakyReLU

    model:add(nn.SpatialConvolution(dimensions[1], 16, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(16))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))

    model:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(16))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))

    model:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))

    model:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(activation())

    model:add(nn.View(32 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    model:add(nn.Linear(32 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 1024))
    model:add(nn.BatchNormalization(1024))
    model:add(activation())
    model:add(nn.Linear(1024, noiseDim))
    --model:add(nn.Dropout(0.2))

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_encoder64(dimensions, noiseDim)
    local model = nn.Sequential()
    local activation = nn.LeakyReLU

    model:add(nn.SpatialConvolution(dimensions[1], 16, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(16))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))

    model:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(16))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))

    model:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))

    model:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))

    model:add(nn.View(32 * 0.25 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    model:add(nn.Linear(32 * 0.25 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 1024))
    model:add(nn.BatchNormalization(1024))
    model:add(activation())
    model:add(nn.Linear(1024, noiseDim))
    --model:add(nn.Dropout(0.2))

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates the decoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_decoder(dimensions, noiseDim)
    local imgSize = dimensions[1] * dimensions[2] * dimensions[3]

    local model = nn.Sequential()
    model:add(nn.Linear(noiseDim, 1024))
    model:add(nn.PReLU())
    model:add(nn.Linear(1024, imgSize))
    model:add(nn.Sigmoid())
    model:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates the decoder part of an upsampling height-16px G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_decoder_upsampling16(dimensions, noiseDim)
    local model = nn.Sequential()
    model:add(nn.Linear(noiseDim, 128))
    model:add(nn.PReLU(nil, nil, true))

    model:add(nn.Linear(128, 512*4*6))
    model:add(nn.BatchNormalization(512*4*6))
    model:add(nn.PReLU(nil, nil, true))
    model:add(nn.View(512, 4, 6))

    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 512, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(512))
    model:add(nn.PReLU(nil, nil, true))

    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 256, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU(nil, nil, true))

    model:add(cudnn.SpatialConvolution(256, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    --model:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates the decoder part of an upsampling height-32px G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_decoder_upsampling32(dimensions, noiseDim)
    local model = nn.Sequential()
    model:add(nn.Linear(noiseDim, 128*8*12))
    model:add(nn.View(128, 8, 12))
    model:add(nn.PReLU(nil, nil, true))

    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(128, 256, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU(nil, nil, true))

    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU(nil, nil, true))

    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_decoder_upsampling32b(dimensions, noiseDim)
    local model = nn.Sequential()
    -- 4x4
    model:add(nn.Linear(noiseDim, 512*4*6))
    model:add(nn.BatchNormalization(512*4*6))
    model:add(nn.PReLU(nil, nil, true))
    model:add(nn.View(512, 4, 6))

    -- 4x4 -> 8x8
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(512))
    model:add(nn.PReLU(nil, nil, true))

    -- 8x8 -> 16x16
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU(nil, nil, true))

    -- 16x16 -> 32x32
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU(nil, nil, true))

    model:add(cudnn.SpatialConvolution(128, dimensions[1], 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.Sigmoid())

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_decoder_upsampling32c(dimensions, noiseDim)
    local model = nn.Sequential()
    -- 4x4
    model:add(nn.Linear(noiseDim, 512*4*6))
    model:add(nn.BatchNormalization(512*4*6))
    model:add(nn.PReLU(nil, nil, true))
    model:add(nn.View(512, 4, 6))

    -- 4x4 -> 8x8
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(512))
    model:add(nn.PReLU(nil, nil, true))

    -- 8x8 -> 16x16
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU(nil, nil, true))

    -- 16x16 -> 32x32
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU(nil, nil, true))
    --model:add(nn.Dropout(0.5))

    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_decoder_upsampling32c_residual(dimensions, noiseDim, cuda)
    --[[
    function createResidual(nbInputPlanes, nbInnerPlanes)
        local seq = nn.Sequential()
        local inner = nn.Sequential()
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInnerPlanes, 1, 1, 1, 1, 0, 0))
        inner:add(nn.SpatialBatchNormalization(32))
        inner:add(nn.LeakyReLU(0.33, true))
        inner:add(cudnn.SpatialConvolution(nbInnerPlanes, nbInnerPlanes, 3, 3, 1, 1, 1, 1))
        inner:add(nn.SpatialBatchNormalization(32))
        inner:add(nn.LeakyReLU(0.33, true))
        inner:add(cudnn.SpatialConvolution(nbInnerPlanes, 256, 1, 1, 1, 1, 0, 0))
        inner:add(nn.SpatialBatchNormalization(256))
        inner:add(nn.LeakyReLU(0.33, true))

        local conc = nn.ConcatTable(2)
        conc:add(inner)
        conc:add(nn.Identity())
        seq:add(conc)
        seq:add(nn.CAddTable())
        return seq
    end
    --]]

    function createResidual(nbInputPlanes, nbInnerPlanes, nbOutputPlanes)
        local seq = nn.Sequential()
        local inner = nn.Sequential()
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInnerPlanes, 3, 3, 1, 1, 1, 1))
        inner:add(nn.SpatialBatchNormalization(nbInnerPlanes))
        inner:add(nn.LeakyReLU(0.33, true))
        inner:add(cudnn.SpatialConvolution(nbInnerPlanes, nbInputPlanes, 3, 3, 1, 1, 1, 1))
        inner:add(nn.SpatialBatchNormalization(nbInputPlanes))
        inner:add(nn.LeakyReLU(0.33, true))

        local conc = nn.ConcatTable(2)
        conc:add(inner)
        conc:add(nn.Identity())
        seq:add(conc)
        seq:add(nn.CAddTable())

        if nbInnerPlanes ~= nbOuterPlanes then
            seq:add(cudnn.SpatialConvolution(nbInnerPlanes, nbOutputPlanes, 1, 1, 1, 1, 0, 0))
            seq:add(nn.SpatialBatchNormalization(nbOutputPlanes))
            seq:add(nn.LeakyReLU(0.33, true))
        end

        return seq
    end

    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 4x6
    model:add(nn.Linear(noiseDim, 512*4*6))
    model:add(nn.BatchNormalization(512*4*6))
    model:add(nn.LeakyReLU(0.33, true))
    model:add(nn.View(512, 4, 6))

    -- 4x6 -> 8x12
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(512))
    model:add(nn.LeakyReLU(0.33, true))
    model:add(createResidual(512, 512, 256))

    -- 8x12 -> 16x24
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(createResidual(256, 256, 128))

    -- 16x24 -> 32x48
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(createResidual(128, 128, 64))

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

-- Creates the decoder part of an upsampling height-32px G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_decoder_upsampling64(dimensions, noiseDim)
    local model = nn.Sequential()
    -- 4x4
    model:add(nn.Linear(noiseDim, 512*4*6))
    model:add(nn.BatchNormalization(512*4*6))
    model:add(nn.PReLU(nil, nil, true))
    model:add(nn.View(512, 4, 6))

    -- 4x4 -> 8x8
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(512))
    model:add(nn.PReLU(nil, nil, true))

    -- 8x8 -> 16x16
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU(nil, nil, true))

    -- 16x16 -> 32x32
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU(nil, nil, true))

    -- 32x32 -> 64x64
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU(nil, nil, true))

    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates G, which is identical to the decoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G(dimensions, noiseDim, cuda)
    if dimensions[2] == 16 then
        return models.create_G_decoder_upsampling16(dimensions, noiseDim, cuda)
    elseif dimensions[2] == 32 then
        return models.create_G_decoder_upsampling32c_residual(dimensions, noiseDim, cuda)
    elseif dimensions[2] == 64 then
        return models.create_G_decoder_upsampling64(dimensions, noiseDim, cuda)
    end
end

-- Creates the G as an autoencoder (encoder+decoder).
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_autoencoder(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if dimensions[2] == 16 then
        model:add(models.create_G_encoder16(dimensions, noiseDim, cuda))
    elseif dimensions[2] == 32 then
        model:add(models.create_G_encoder32(dimensions, noiseDim, cuda))
    else
        model:add(models.create_G_encoder64(dimensions, noiseDim, cuda))
    end

    if dimensions[2] == 16 then
        model:add(models.create_G_decoder_upsampling16(dimensions, noiseDim, cuda))
    elseif dimensions[2] == 32 then
        model:add(models.create_G_decoder_upsampling32c_residual(dimensions, noiseDim, cuda))
    else
        model:add(models.create_G_decoder_upsampling64(dimensions, noiseDim, cuda))
    end

    return model
end

-- Creates D.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_D(dimensions, cuda)
    --[[
    if dimensions[2] == 16 then
        return models.create_D16b(dimensions, cuda)
    else
        return models.create_D32e(dimensions, cuda)
    end
    --]]
    --return models.create_D32_st3(dimensions, cuda)
    if dimensions[2] == 16 then
        print("Picking D 16")
        return models.create_D16(dimensions, cuda)
    elseif dimensions[2] == 32 then
        print("Picking D 32f")
        return models.create_D32f(dimensions, cuda)
    else
        print("Picking D 64")
        return models.create_D64(dimensions, cuda)
    end
end

-- Create base D network for 32x32 images.
-- Color and grayscale currently get the same network, which is suboptimal.
-- @param dimensions Image dimensions as table of {count channels, height, width}
-- @returns Sequential
function models.create_D64(dimensions, cua)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 64x96
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 32x48
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 16x24
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 8x12
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 4x6
    conv:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.5))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 2x3
    conv:add(nn.View(512 * 2 * 3))
    conv:add(nn.Linear(512 * 2 * 3, 512))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D16(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 16x24
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    --conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 16x24
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))

    -- strided
    conv:add(nn.SpatialConvolution(128, 256, 5, 5, 2, 2, (5-1)/2))
    conv:add(nn.PReLU(nil, nil, true))

    -- 8x12
    conv:add(nn.SpatialConvolution(256, 512, 5, 5, 1, 1, (5-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 4x6
    conv:add(nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout())

    -- 4x6
    conv:add(nn.View(1024 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(1024 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 128))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(128, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x48
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    --conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 32x48
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))

    -- strided
    conv:add(nn.SpatialConvolution(128, 256, 5, 5, 2, 2, (5-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))

    -- 16x24
    conv:add(nn.SpatialConvolution(256, 512, 5, 5, 1, 1, (5-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x12
    conv:add(nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout())
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 4x6
    conv:add(nn.View(1024 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(1024 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 128))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(128, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32b(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x48
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x24
    conv:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x12
    conv:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 4x6
    conv:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))

    -- 4x6
    conv:add(nn.View(32 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(32 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 128))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(128, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32c(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x48
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 16x24
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))

    -- strided
    conv:add(nn.SpatialConvolution(128, 256, 5, 5, 2, 2, (5-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))

    -- 8x12
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))

    -- strided
    conv:add(nn.SpatialConvolution(512, 512, 5, 5, 2, 2, (5-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))

    -- 4x6
    conv:add(nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))

    -- strided
    conv:add(nn.SpatialConvolution(1024, 1024, 5, 5, 2, 2, (5-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.5))

    -- 2x3
    conv:add(nn.View(1024 * 0.25 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(1024 * 0.25 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 512))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32d(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x48
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout(0.5))
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))

    -- max
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialDropout(0.5))

    -- 16x24
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout(0.5))
    conv:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))

    local conv2 = nn.Sequential()
    -- max
    conv2:add(nn.SpatialMaxPooling(2, 2))
    conv2:add(nn.SpatialDropout(0.5))

    -- 8x12
    conv2:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2))
    conv2:add(nn.PReLU(nil, nil, true))
    conv2:add(nn.SpatialDropout(0.5))
    conv2:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, (3-1)/2))
    --conv2:add(nn.SpatialBatchNormalization(512))
    conv2:add(nn.PReLU(nil, nil, true))

    -- max
    conv2:add(nn.SpatialMaxPooling(2, 2))
    conv2:add(nn.SpatialDropout(0.5))
    --conv2:add(nn.Reshape(512 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))

    -- 4x6
    conv2:add(nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, (3-1)/2))
    conv2:add(nn.PReLU(nil, nil, true))
    conv2:add(nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, (3-1)/2))
    conv2:add(nn.PReLU(nil, nil, true))

    -- max
    conv2:add(nn.SpatialMaxPooling(2, 2))
    conv2:add(nn.SpatialDropout(0.5))
    conv2:add(nn.Reshape(1024 * 0.25 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))


    local conv3 = nn.Sequential()
    --conv3:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    conv3:add(nn.SpatialMaxPooling(2, 2))
    conv3:add(nn.Reshape(256 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv3:add(nn.Linear(256 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 64))
    --conv3:add(nn.BatchNormalization(64))
    conv3:add(nn.PReLU(nil, nil, true))

    local concat = nn.Concat(2)
    concat:add(conv2)
    concat:add(conv3)
    conv:add(concat)

    -- 2x3
    conv:add(nn.Linear(64 + (1024 * 0.25 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]), 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32e(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x48
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 16x24
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 8x12
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 4x6
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.3))
    --conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 2x3
    conv:add(nn.View(512*4*6))
    conv:add(nn.Linear(512*4*6, 64))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Linear(64, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D32f(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x48
    conv:add(cudnn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.33, true))
    conv:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.33, true))
    conv:add(nn.SpatialDropout(0.2))

    -- 16x24
    conv:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.33, true))
    conv:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.33, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 8x12
    conv:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.33, true))
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 4x6
    conv:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.33, true))
    conv:add(nn.SpatialDropout(0.3))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 2x3
    conv:add(nn.View(512*4*6))
    conv:add(nn.Linear(512*4*6, 64))
    conv:add(nn.LeakyReLU(0.33, true))
    conv:add(nn.Linear(64, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D64f(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 64x96
    conv:add(cudnn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 32x48
    conv:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 16x24
    conv:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 8x12
    conv:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.3))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 4x6
    conv:add(nn.View(512*4*6))
    conv:add(nn.Linear(512*4*6, 64))
    conv:add(nn.PReLU(nil, nil, true))
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
