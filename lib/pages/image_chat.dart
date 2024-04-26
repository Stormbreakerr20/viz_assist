import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:dash_chat_2/dash_chat_2.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:http_parser/http_parser.dart' as http_parser;

class ChatPage extends StatefulWidget {
  const ChatPage(
      {Key? key, required this.imageFile, required this.initialMessage})
      : super(key: key);
  final File imageFile;
  final String initialMessage;

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  List<ChatMessage> messages = [];

  ChatUser currentUser = ChatUser(id: '0', firstName: 'Me');
  ChatUser queryBot = ChatUser(
      id: '1',
      firstName: 'VizAssist',
      profileImage:
          'https://w1.pngwing.com/pngs/278/853/png-transparent-line-art-nose-chatbot-internet-bot-artificial-intelligence-snout-head-smile-black-and-white.png');
  ChatMessage initialChatMessage = ChatMessage(
    text: 'Hello! How can I help you today?',
    user: ChatUser(id: '1', firstName: 'VizAssist'),
    createdAt: DateTime.now(),
  );

  void initState() {
    super.initState();
    _initializeChat();
  }

  Future<void> _initializeChat() async {
    // Create the initial message
    ChatMessage initialMessage = ChatMessage(
      text: widget.initialMessage,
      user: currentUser,
      createdAt: DateTime.now(),
    );

    // Add the initial message to the list
    setState(() {
      messages.insert(0, initialMessage);
    });

    // Check if the message contains a file
    File? file = widget.imageFile;

    try {
      // Send the API request with the initial message and file (if any)
      String? responseText = await _sendApiRequest(initialMessage, file);
      print("responseText: $responseText");

      // Create a response message based on the API response
      ChatMessage responseMessage = ChatMessage(
        text: responseText ?? "No response from API",
        user: queryBot,
        createdAt: DateTime.now(),
      );

      // Add the response message to the list
      setState(() {
        messages.insert(0, responseMessage);
      });
    } catch (error) {
      print("Error sending initial message: $error");
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'VizAssist',
      theme: ThemeData.light(), // Apply the dark theme
      home: Scaffold(
        appBar: AppBar(
          centerTitle: true,
          title: const Text('VizAssist: Chat'),
        ),
        body: _buildUI(),
      ),
    );
  }

  Widget _buildUI() {
    return DashChat(
        inputOptions: InputOptions(trailing: [
          IconButton(
            icon: const Icon(Icons.mic),
            onPressed: _listen,
          ),
        ]),
        currentUser: currentUser,
        onSend: _sendMessage,
        messages: messages);
  }

  void _sendMessage(ChatMessage message) async {
    // Get the current user's message
    ChatMessage userMessage = ChatMessage(
      text: message.text,
      user: currentUser,
      createdAt: DateTime.now(),
    );

    // Add the user's message to the list
    setState(() {
      messages.insert(0, userMessage);
    });

    // Check if the message contains a file
    File? file = widget.imageFile;

    // Send the API request with the message and file (if any)
    String? responseText;
    responseText = await _sendApiRequest(userMessage, file);
    print("responseText: $responseText");

    // Create a response message and add it to the list
    ChatMessage responseMessage = ChatMessage(
      text: responseText,
      user: queryBot, // Assuming the API response is from a bot user with id 0
      createdAt: DateTime.now(),
    );

    setState(() {
      messages.insert(0, responseMessage);
    });
  }

  Future<String> _sendApiRequest(ChatMessage chatMessage, File file) async {
    var uri = Uri.parse('https://dfa9-34-139-66-56.ngrok-free.app/chat');
    uri = uri.replace(queryParameters: {
      'prompt': chatMessage.text,
    });
    var request = http.MultipartRequest('POST', uri);
    request.files.add(http.MultipartFile.fromBytes(
        'file', file.readAsBytesSync(),
        filename: file.path.split('/').last));
    var streamedResponse = await request.send();
    var res = await http.Response.fromStream(streamedResponse);
    var responseBody = json.decode(res.body);
    String text = responseBody['content'];
    if (res.statusCode == 200) {
      print("Uploaded!");
      print("Response: $text");
      return text;
    } else {
      print("Failed to upload");
      // print error
      print("Server response: $res");
    }
    return 'text';
  }

  // function for speech to text
  void _listen() async {
    final stt.SpeechToText speech = stt.SpeechToText();
    bool available = await speech.initialize(
      onStatus: (val) => print('onStatus: $val'),
      onError: (val) => print('onError: $val'),
    );
    if (available) {
      speech.listen(
        onResult: (val) => setState(() {
          messages.insert(
              0,
              ChatMessage(
                text: val.recognizedWords,
                user: currentUser,
                createdAt: DateTime.now(),
              ));
        }),
      );
    } else {
      print("The user has denied the use of speech recognition.");
    }
  }
}