require 'image'
require 'paths'

stacks = {}

stacks.path = 'stacks.t7'
--stack02
stacks.final = torch.Tensor(30,1,500,500)

--stack04/06, four flips two rotations data augmentation
stacks.input = torch.Tensor(1,1,500,500)
stacks.output = torch.Tensor(1,1,500,500)

function stacks.load(override)
    override = override or false
    if not paths.filep(stacks.path) or override then
        local augfactor = 2*2*2 --rotate, vflip, hflip

        local s4size = 30
        local s6size = 30
        local s2size = 30
        local s7size = 80
        local s7crops = 5  --torch does 5 kinds of crops

        local total = augfactor * (s4size + s6size + (s7size * s7crops))
        --local total = augfactor * (s4size + s6size)

        local index = 1
        local perm = torch.randperm(total)

        stacks.input = torch.FloatTensor(total, 1, 500, 500)
        stacks.output = torch.FloatTensor(total, 500, 500)

        print('loading ' .. total .. ' samples')

        local function add_pair(t1, t2) 
            for rot = 1,2 do
                for rot = 1, 2 do
                    for rot = 1, 2 do
                        --add using permutation matrix
                        local reali = perm[index]
                        index = index + 1
                        stacks.input[reali][1] = t1
                        stacks.output[reali] = t2

                        t1 = image.hflip(t1)
                        t2 = image.hflip(t2)
                    end
                    t1 = image.vflip(t1)
                    t2 = image.vflip(t2)
                end
                t1 = image.rotate(t1,math.pi/2)
                t2 = image.rotate(t2,math.pi/2)
            end
        end


        --load stack 4
        for x = 1, s4size do
            local loadi = x - 1
            local t1 = image.load('data/stack04/04_raw-' .. loadi .. '.png')
            local t2 = image.load('data/stack04/04_lbl-' .. loadi .. '.png')

            add_pair(t1, t2)
        end

        --load stack 6
        for x = 1, s6size do
            local loadi = x - 1
            local t1 = image.load('data/stack06/06_raw-' .. loadi .. '.png')
            local t2 = image.load('data/stack06/06_lbl-' .. loadi .. '.png')

            add_pair(t1, t2)
        end

        --load stack 7
        for x = 1, s7size do
            local loadi = x - 1
            local t1 = image.load('data/stack07/07_raw-' .. loadi .. '.png')
            local t2 = image.load('data/stack07/07_lbl-' .. loadi .. '.png')

            local crops = {'c', 'tl', 'tr', 'bl', 'br'}

            for _,c in ipairs(crops) do 
                local c1 = image.crop(t1, c, 500, 500)
                local c2 = image.crop(t2, c, 500, 500)
                add_pair(c1, c2)
            end
        end

        --load stack 2
        for x = 1, s2size do
            local loadi = x - 1
            local t1 = image.load('data/stack02/02_raw-' .. loadi .. '.png')
            stacks.final[x] = t1
        end
    else
        stacks = torch.load(stacks.path)
    end
end

function stacks.save() 
    torch.save(stacks.path, stacks)
end
