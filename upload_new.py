import http.server
import socketserver
import os
import cgi
import urllib.parse

PORT = 8000
UPLOAD_DIR = "uploads"

class UploadHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve the upload form and directory listing
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            # Create uploads directory if it doesn't exist
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)
            
            # Generate HTML response
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>File Upload Server</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .file-list { margin-top: 20px; }
                    .upload-form { margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <h1>File Upload Server</h1>
                <div class="upload-form">
                    <form enctype="multipart/form-data" method="post">
                        <input type="file" name="file" multiple>
                        <input type="submit" value="Upload">
                    </form>
                </div>
                <div class="file-list">
                    <h2>Uploaded Files:</h2>
                    <ul>
            """
            
            # List uploaded files
            for file in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, file)
                if os.path.isfile(file_path):
                    encoded_file = urllib.parse.quote(file)
                    html += f'<li><a href="/{UPLOAD_DIR}/{encoded_file}">{file}</a></li>'
            
            html += """
                    </ul>
                </div>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode())
        else:
            # Serve files from the uploads directory
            super().do_GET()

    def do_POST(self):
        # Handle file upload
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            # Get the file(s) from the form
            files = form['file'] if isinstance(form['file'], list) else [form['file']]
            
            # Create uploads directory if it doesn't exist
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)
            
            for file_item in files:
                if file_item.filename:
                    # Get the file name and save path
                    file_name = os.path.basename(file_item.filename)
                    file_path = os.path.join(UPLOAD_DIR, file_name)
                    
                    # Save the file
                    with open(file_path, 'wb') as output_file:
                        output_file.write(file_item.file.read())
            
            # Redirect back to the main page
            self.send_response(303)
            self.send_header('Location', '/')
            self.end_headers()
            
        except Exception as e:
            self.send_error(500, f"Error processing upload: {str(e)}")

if __name__ == '__main__':
    # Set up the server
    with socketserver.TCPServer(("", PORT), UploadHandler) as httpd:
        print(f"Serving at port {PORT}")
        print(f"Upload directory: {os.path.abspath(UPLOAD_DIR)}")
        print("Visit http://localhost:8000 to access the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")
            httpd.server_close()