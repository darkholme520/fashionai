import numpy as np
import cv2

def image_resize_and_pad(img_orgi):# TODO add adjustable shape, shape = (224,224,3)):
    scale = 224 / max(img_orig.shape)
    s1 = int(round(img_orig.shape[0]) *scale)
    s2 = int(round(img_orig.shape[1]) *scale)
    if img_orig.shape[0] == 224 and img_orig.shape[1] == 224:
        img_resi = img_orig
        offset = [0,0,0]
    else:
        img_resi = cv2.resize(img_orig,(s1,s2))
        pad = [224,224,0]
        padsplit = []
        for i in range(len(pad)):
            pad[i] -= img_resi.shape[i]
            if pad[i] < 0:
                pad[i] = 0
            padsplit.append([0,0])
            padsplit[i][0] = int(pad[i]/2.)
            padsplit[i][1] = pad[i] - padsplit[i][0]

        img_resi = np.pad(img_resi, padsplit, 'constant', constant_values = 0)
    
    print(img_resi.shape)
    assert(img_resi.shape[0] == 224)
    assert(img_resi.shape[1] == 224)
    assert(img_resi.shape[2] == 3)
    return img_resi

def image_normalization(img_resi):
    ''' matlab code:
    pixel_means = reshape([102.9801, 115.9465, 122.7717], [1 1 3]); ## ?
    img_stan = single(img_resi); ## ?
    img_stan = permute(img_stan, [2, 1, 3]); ## done in np.swapaxes
    img_stan = img_stan(:, :, [3, 2, 1]); ## no need to do
    img_stan = bsxfun(@minus, img_stan, pixel_means); ## ?
    '''
    img_stan = np.swapaxes(img_resi,0,1)
    return img_stan


def pipeline_forword(img_orig, pipeline):
    ### image resize & pad
    img_resi = image_resize_and_pad(img_orig)
    
    ## image normalization
    img_stan = image_normalization(img_resi)
    
    
    
    get_orig_coordinate = @(p)((p+0.5)*224-repmat([offset(2),offset(1)]',[pipeline.num_points,1]))/scale
    visibility_case = {'Visible','Occlude','Inexistent'}
    
    ### stage 1 fp
    res_stage1 = pipeline['net_stage1'].forward({img_stan})
    landmark_stage1 = res_stage1{1}(1:pipeline['num_points']*2)
    
    [~,v1] = max(reshape(res_stage1{1}(1+pipeline['num_points']*2:end),3,pipeline['num_points']))
    visibility_stage1 = visibility_case(v1)
    
    prediction_stage1 = struct('landmark',get_orig_coordinate(landmark_stage1),...
        'visibility',{visibility_stage1})
    
    ### stage 2 fp   
    res_stage2 = pipeline.net_stage2.forward({img_stan,landmark_stage1})
    landmark_stage2 = landmark_stage1-res_stage2{1}(1:pipeline.num_points*2)/5
    
    [~,v2] = max(reshape(res_stage2{1}(1+pipeline.num_points*2:end),3,pipeline.num_points))
    visibility_stage2 = visibility_case(v2)
    
    prediction_stage2 = struct('landmark',get_orig_coordinate(landmark_stage2),...
        'visibility',{visibility_stage2})
    
    ### stage 3 fp
    res_stage3_easy = pipeline.net_stage3_easy.forward({img_stan,landmark_stage2})
    res_stage3_hard = pipeline.net_stage3_hard.forward({img_stan,landmark_stage2})
    landmark_stage3 = landmark_stage2 -...
        (res_stage3_easy{1}(1:pipeline.num_points*2)/5 ...
        + res_stage3_hard{1}(1:pipeline.num_points*2)/5)/2
    
    [~,v3] = max(reshape(res_stage3_easy{1}(1+pipeline.num_points*2:end)+...
        res_stage3_hard{1}(1+pipeline.num_points*2:end),3,pipeline.num_points))
    visibility_stage3 = visibility_case(v3)
    
    prediction_stage3 = struct('landmark',get_orig_coordinate(landmark_stage3),...
        'visibility',{visibility_stage3})
    
    ### output
    
    prediction = struct('stage1',prediction_stage1,...
                        'stage2',prediction_stage2,...
                        'stage3',prediction_stage3,...
                        'num_points',pipeline.num_points)
    return prediction

