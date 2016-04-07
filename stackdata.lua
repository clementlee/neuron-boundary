require 'image'
require 'paths'

stacks = {}

stacks.path = 'stacks.t7'
--stack02
stacks.final = torch.Tensor(30,1,500,500)

--stack04/06, four flips four data augmentation
stacks.input = torch.Tensor(1,1,500,500)
stacks.output = torch.Tensor(1,1,500,500)

function stacks.load(override, save)
    override = override or false
    save = save or false
    if not paths.filep(stacks.path) or override then
        local augfactor = 4*2*2 --rotate, vflip, hflip

        local s4size = 30
        local s6size = 30
        local s2size = 30
        --local s7size = 80
        --local s7crops = 551 * 551 --possible crops of 1050x1050

        --local total = augfactor * (s4size + s6size + (s7size * s7crops))
        local total = augfactor * (s4size + s6size)

        local index = 1
        local perm = torch.randperm(total)

        stacks.input = torch.Tensor(total, 1, 500, 500)
        stacks.output = torch.Tensor(total, 500, 500)

        print('loading ' .. total .. ' samples')

        function add_pair(t1, t2) 
            --binarize output
            t2:apply(function(x)
                if x == 0 then
                    return x
                else
                    return 1
                end
            end)

            for rot = 1,4 do
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
        
        --load stack 2
        for x = 1, s2size do
            local loadi = x - 1
            local t1 = image.load('data/stack02/02_raw-' .. loadi .. '.png')
            stacks.final[x] = t1
        end

        --save
        if save then
            torch.save(stacks.path, stacks)
        end
    else
        stacks = torch.load(stacks.path)
    end
end
