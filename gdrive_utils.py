import requests
import os 

def download_file_from_google_drive(id, destination):
    """
    id: get the id from google drive file share link 
        (link = https://drive.google.com/file/d/1NHr8JCPBHZ3QHrffD26H7UyaaKuKHddH/view?usp=sharing 
         if = 1NHr8JCPBHZ3QHrffD26H7UyaaKuKHddH)
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
        
dict_file_id = {
    "data_kode_wilayah.csv": "1cpEWjWNbivauHrTHnVq_rEVm1B9kow0w",

    "faster_rcnn_R_101_FPN_3x_SIM.pth" : "1ib9t16yykaDdMFLewE__jvwJ9kDEHKIP", #old
    "box_detector_SIM__faster_rcnn_R_101_FPN_3x.pth" : "1E3en7ixhzjZaDF7-57WWpJzQeE8k9mER",
    
    "faster_rcnn_R_101_FPN_3x_KTP.pth" : '1hG1aS5tvpvdvGJMoO5BhDViNt17mX3W-', #old
    "box_detector_KTP__faster_rcnn_R_101_FPN_3x.pth" : '1I_LdCq64GGdtqrP6WeNBolHr7LizkvbZ',
    

    # nomor and expired date
    "faster_rcnn_R_101_FPN_3x_NIK_NOSIM.pth" : "1U8DTSN-ldmZlmEYxZmz29eOYmPCCS_5s", #old
    "char_detector_NIK_NOSIM_acc98__X101_custom_mapper_aug.pth" : "1owaegmLmZiLMpYXIrXmyEkTIYJesW8a3",
    "char_detector_NIK_NOSIM_EXP_acc933__X101_custom_mapper_aug.pth" : "1vncm8D_JjtrX9D0rrTe4XVxMiNCWe3ch",
    "char_detector_NIK_NOSIM_EXP_acc946__X101_custom_mapper_aug.pth" : "1yHAi6zNcqSOzs8VM8pces0ysl0WK2AxH",
    
    # nama
    "model_final_ktp_sim-fix_rgb_rotated_X101_custom_mapper_many_augmentation.pth" : "1n4K4kfS90EZ8_G71ixWjgiIujguYI2Jk", 
    "model_final_ktp_sim-fix_rgb_rotated_X101_custom_mapper.pth" : "1MRnt4Oyz2DFb3TSU2hhSZdPRmSVAVIwG",
    "model_final_ktp_sim-fix_rgb_x101.pth" : "1ixH6guUmxnLcy11QBiYbe1fNQhT9K3lx", 
    "model_final_ktp_sim-grayscale.pth" : "1791kG9LbcorDX-Wb-kZ38shib4x-Jkev"
    
}
                
def download_file(file_path):
    if not os.path.exists(file_path):
        filename = os.path.basename(file_path)
        if filename in dict_file_id.keys():
            file_id = dict_file_id[filename]
            print(f"DOWNLOADING {filename} ......")
            download_file_from_google_drive(file_id, file_path)
        else:
            print(f"FILE {filename} IS NOT AVAILABLE")
    else:
        pass
        # print("{} is already downloaded".format(file_path))