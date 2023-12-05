from furhat_remote_api import FurhatRemoteAPI
import time

#########################
#   INIT
#########################

furhat = FurhatRemoteAPI("localhost")

# Get the voices on the robot
voices = furhat.get_voices()

# Set the voice of the robot
furhat.set_voice(name='Matthew')

# Say "Hi there!"
furhat.say(text="Hi there!")
#---------------------------------------------------

def to_LLM(msg):
    return f"I am not ready yet, you said: {msg}"

def conv_loop():
    while(True):
        res = furhat.listen()

        print("RES", res.message) #string
        #print(type(res)) # <class 'swagger_client.models.status.Status'>

        user_input = res.message

        if(user_input.lower() == 'die'):
            break

        response = to_LLM(user_input)

        furhat.say(text=response)
        time.sleep(0.5)


conv_loop()


