from tflite_support import metadata

populator_dst = metadata.MetadataPopulator.with_model_file('/content/gdrive/MyDrive/LPD_SS2/tfLite/bestEF1_edgetpu.tflite')

with open('/content/gdrive/MyDrive/LPD_SS2/tfLite/bestEF1.tflite', 'rb') as f:
  populator_dst.load_metadata_and_associated_files(f.read())

populator_dst.populate()
updated_model_buf = populator_dst.get_model_buffer()