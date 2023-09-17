css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/Stick_Figure.svg/1451px-Stick_Figure.svg.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

summarization_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/previews/005/155/279/non_2x/sigma-greek-symbol-capital-letter-uppercase-font-icon-black-color-illustration-flat-style-image-vector.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

answer_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/previews/005/155/279/non_2x/sigma-greek-symbol-capital-letter-uppercase-font-icon-black-color-illustration-flat-style-image-vector.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
sources_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/previews/005/155/279/non_2x/sigma-greek-symbol-capital-letter-uppercase-font-icon-black-color-illustration-flat-style-image-vector.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

