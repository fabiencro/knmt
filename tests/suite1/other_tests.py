# def test_double(gpu = None):
#     print "test_double", "gpu:", gpu
#     Vi = 1000 + 2
#     Vo = 1000 + 2
#     Ei = 620
#     Eo = 620
#     Hi = 1000
#     Ho = 1000
#     Ha = 1000
#     Hl = 500
#
#     def gen_data_double(min_length, max_length, batch_size):
#         training_data_initial = [np.random.randint(2,Vi - 1, size = (sz,)).astype(np.int32)
#                     for sz in np.random.randint(min_length,max_length, size = (batch_size,))]
#         training_data_double = [(a,a[::-1]) for a in training_data_initial]
#         return training_data_double
#
#
#     used_make_batch = make_batch_fit2
#     used_encdec = EncDecFitMask
#
#     valid_data_double = gen_data_double(20, 50, 30)
#     valid_src_batch, valid_tgt_batch, valid_src_mask = used_make_batch(valid_data_double, gpu = gpu)
#
#     encdec = used_encdec(Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl)
#
#     if gpu is not None:
#         encdec = encdec.to_gpu(gpu)
#
#     def compute_valid():
#         loss, attn = encdec(valid_src_batch, valid_tgt_batch, valid_src_mask)
#         for i in xrange(len(attn)):
#             print attn[i].data[:, i]
#             print "argmax", np.argmax(attn[i].data,1)
#         print loss.data
#
#
#     def minibatch_provider(mb_size, nb_mb_for_sorting):
#         while 1:
#             training_data_double = gen_data_double(20, 50, mb_size * nb_mb_for_sorting)
#             training_data_double.sort(key = lambda x:len(x[1]))
#             for num_batch in xrange(nb_mb_for_sorting):
#                 mb_raw = training_data_double[num_batch * mb_size : num_batch*mb_size + mb_size]
#                 src_batch, tgt_batch, src_mask = used_make_batch(mb_raw, gpu = gpu)
#                 yield src_batch, tgt_batch, src_mask
#
#     optimizer = optimizers.AdaDelta()
#     optimizer.setup(encdec)
#
#     mb_provider = minibatch_provider(80, 20)
#
#     def train_once_random_data():
# #         training_data_double = gen_data_double(20, 50, 100)
# #         src_batch, tgt_batch = make_batch2(training_data_double, gpu = gpu)
# #         for i in range(len(src_batch)):
# #             print i, type(src_batch[i].data), type(tgt_batch[i].data)
#         src_batch, tgt_batch, src_mask = mb_provider.next()
#
#         encdec.zerograds()
#         loss, attn = encdec(src_batch, tgt_batch, src_mask)
#         print loss.data
#         loss.backward()
#         optimizer.update()
#
#     try:
#         for i in xrange(10000):
#             print i,
#             train_once_random_data()
#             if i%100 == 0:
#                 print "valid",
#                 compute_valid()
#             if i%200 == 0:
#                 print "sample"
#                 src_batch, tgt_batch, src_mask = mb_provider.next()
#                 sample, score = encdec(src_batch, 50, src_mask, use_best_for_sample = True)
#                 for i in range(len(src_batch)):
#                     print i
#                     print src_batch[i].data
#                     print sample[i]
# #                 print sample
# #                 print score
#     finally:
#         log.info("saving model to ")
#         serializers.save_npz("encdec_model_final", encdec)
