import caffe

def pipeline_init( model_type , poj_path):

    if model_type == 'upper':
        num_points = 6

        model_stage1 = poj_path + '/fashion-landmarks/models/FLD_upper/stage1.prototxt'
        weights_stage1 = poj_path + '/fashion-landmarks/models/FLD_upper/stage1.caffemodel'

        model_stage2 = poj_path + '/fashion-landmarks/models/FLD_upper/cascade.prototxt'
        weights_stage2 = poj_path + '/fashion-landmarks/models/FLD_upper/stage2.caffemodel'

        model_stage3 = poj_path + '/fashion-landmarks/models/FLD_upper/cascade.prototxt'
        weights_stage3_easy = poj_path + '/fashion-landmarks/models/FLD_upper/stage3_easy.caffemodel'
        weights_stage3_hard = poj_path + '/fashion-landmarks/models/FLD_upper/stage3_hard.caffemodel'
    
    elif model_type == 'lower':
        num_points = 4
    
        model_stage1 = poj_path + '/fashion-landmarks/models/FLD_lower/stage1.prototxt'
        weights_stage1 = poj_path + '/fashion-landmarks/models/FLD_lower/stage1.caffemodel'

        model_stage2 = poj_path + '/fashion-landmarks/models/FLD_lower/cascade.prototxt'
        weights_stage2 = poj_path + '/fashion-landmarks/models/FLD_lower/stage2.caffemodel'

        model_stage3 = poj_path + '/fashion-landmarks/models/FLD_lower/cascade.prototxt'
        weights_stage3_easy = poj_path + '/fashion-landmarks/models/FLD_lower/stage3_easy.caffemodel'
        weights_stage3_hard = poj_path + '/fashion-landmarks/models/FLD_lower/stage3_hard.caffemodel'
    
    elif model_type == 'full':
        num_points = 8
    
        model_stage1 = poj_path + '/fashion-landmarks/models/FLD_full/stage1.prototxt'
        weights_stage1 = poj_path + '/fashion-landmarks/models/FLD_full/stage1.caffemodel'

        model_stage2 = poj_path + '/fashion-landmarks/models/FLD_full/cascade.prototxt'
        weights_stage2 = poj_path + '/fashion-landmarks/models/FLD_full/stage2.caffemodel'

        model_stage3 = poj_path + '/fashion-landmarks/models/FLD_full/cascade.prototxt'
        weights_stage3_easy = poj_path + '/fashion-landmarks/models/FLD_full/stage3_easy.caffemodel'
        weights_stage3_hard = poj_path + '/fashion-landmarks/models/FLD_full/stage3_hard.caffemodel'

    else:
        print('Undefiened Model Type')

    #caffe.reset_all()
    caffe.set_mode_gpu()
    caffe.set_device(0)
    caffe.set_multiprocess(True)

    # create net and load weights
    net_stage1 = caffe.Net(model_stage1, weights_stage1, caffe.TEST) 
    net_stage2 = caffe.Net(model_stage2, weights_stage2, 'test')
    net_stage3_easy = caffe.Net(model_stage3, weights_stage3_easy, 'test')
    net_stage3_hard = caffe.Net(model_stage3, weights_stage3_hard, 'test')

    pipeline = {
        'num_points':num_points,
        'net_stage1':net_stage1,
        'net_stage2':net_stage2,
        'net_stage3_easy':net_stage3_easy,
        'net_stage3_hard':net_stage3_hard
    }

    return pipeline

if __name__ == '__main__':
    poj_path = '/home/cjl/fashionai'
    testppl = pipeline_init('upper', poj_path)
    print(testppl)