from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
import ccxt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import asyncio
import json
import pytz

# Token bot từ BotFather
TOKEN = "8081244500:AAFkXKLfVoXQeqDYVW_HMdXluGELf9AWD3M"

# Địa chỉ Webhook (thay YOUR_RENDER_URL bằng URL ứng dụng Render của bạn)
WEBHOOK_URL = f"https://telegrambot-an3l.onrender.com"
# Khởi tạo KuCoin
exchange = ccxt.kucoin()
# Lưu trữ lịch sử tín hiệu
signal_history = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gửi tin nhắn chào mừng và hướng dẫn."""
    await update.message.reply_text(
        "Chào mừng! Tôi là bot phân tích kỹ thuật của anh Hưng Thạnh đẹp trai.\n"
        "Dưới đây là các lệnh bạn có thể sử dụng:\n"
        "Gõ /chart <mã giao dịch> để xem biểu đồ kỹ thuật (ví dụ: /chart BTC/USDT).\n"
        "Gõ /top để xem top 10 cặp giao dịch tăng và giảm mạnh nhất trong 1 giờ qua.\n"
        "Gõ /signal <mã giao dịch> để nhận tín hiệu mua bán và lưu lịch sử.\n"
        "Gõ /cap <mã giao dịch> để xem thông tin giá hiện tại.\n"
        "Gõ /subscribe để đăng ký nhận thông báo tự động.\n"
        "Gõ /unsubscribe để hủy đăng ký nhận thông báo."
    )


# File lưu danh sách subscribers
SUBSCRIBERS_FILE = "subscribers.json"

def load_subscribers():
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, "r") as f:
            return json.load(f)
    return []

def save_subscribers(subscribers):
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(subscribers, f)

subscribers = load_subscribers()

# Khởi tạo múi giờ Việt Nam
vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Đăng ký nhận thông báo tự động."""
    chat_id = update.message.chat_id
    if chat_id not in subscribers:
        subscribers.append(chat_id)
        save_subscribers(subscribers)
        await update.message.reply_text("Bạn đã đăng ký nhận thông báo tự động!")
    else:
        await update.message.reply_text("Bạn đã đăng ký trước đó rồi.")

async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Huỷ đăng ký nhận thông báo tự động."""
    chat_id = update.message.chat_id
    if chat_id in subscribers:
        subscribers.remove(chat_id)
        save_subscribers(subscribers)
        await update.message.reply_text("Bạn đã huỷ đăng ký nhận thông báo.")
    else:
        await update.message.reply_text("Bạn chưa đăng ký trước đó.")

async def current_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol = context.args[0] if context.args else None
        if not symbol:
            await update.message.reply_text("Vui lòng cung cấp mã giao dịch. Ví dụ: /cap BTC/USDT")
            return

        markets = exchange.load_markets()
        if symbol not in markets:
            await update.message.reply_text(f"Mã giao dịch không hợp lệ: {symbol}. Vui lòng kiểm tra lại.")
            return

        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        percentage_change = ticker['percentage']
        volume_24h = ticker.get('quoteVolume', 0)  # Khối lượng giao dịch trong 24 giờ qua

        timestamp = (
            pd.to_datetime(ticker['timestamp'], unit='ms')
            .tz_localize('UTC')
            .tz_convert(vietnam_tz)
            .strftime('%Y-%m-%d %H:%M:%S')
        )

        message = (
            f"Thông tin giá hiện tại cho {symbol}:\n"
            f"- Giá hiện tại: {current_price:.2f} USD\n"
            f"- Biến động trong 24 giờ qua: {percentage_change:.2f}%\n"
            f"- Khối lượng giao dịch trong 24 giờ qua: {volume_24h:.2f} USD\n"
            f"- Thời gian cập nhật: {timestamp}"
        )
        await update.message.reply_text(message)

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")





async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Tạo và gửi biểu đồ kỹ thuật."""
    try:
        symbol = context.args[0] if context.args else context.chat_data.get("symbol")
        if not symbol:
            await update.message.reply_text("Vui lòng cung cấp mã giao dịch. Ví dụ: /chart BTC/USDT")
            return

        timeframe = '1h'
        limit = 200

        markets = exchange.load_markets()
        if symbol not in markets:
            await update.message.reply_text(f"Mã giao dịch không hợp lệ: {symbol}. Vui lòng kiểm tra lại.")
            return

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Chuyển đổi timestamp sang giờ Việt Nam
        df['timestamp'] = (
            pd.to_datetime(df['timestamp'], unit='ms')
            .dt.tz_localize('UTC')
            .dt.tz_convert(vietnam_tz)
        )


        # Tính toán các chỉ báo kỹ thuật
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA100'] = df['close'].rolling(window=100).mean()


        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - df['close'].rolling(window=20).std()


        # MACD
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']


        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))


       # Biểu đồ Candlestick và MACD được đặt riêng biệt
        fig = make_subplots(
            rows=4,  # Tăng số lượng hàng lên 4 để tách MACD khỏi biểu đồ giá
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.2, 0.2, 0.1],  # Cập nhật chiều cao từng hàng
            specs=[[{"secondary_y": True}], [{}], [{}], [{}]]
        )

        # Candlestick và Bollinger Bands
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Candlestick"
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['BB_Upper'],
            mode='lines',
            line=dict(color='red', width=1),
            name='BB Upper'
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['BB_Middle'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='BB Middle'
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['BB_Lower'],
            mode='lines',
            line=dict(color='green', width=1),
            name='BB Lower'
        ), row=1, col=1, secondary_y=False)

        # Biểu đồ khối lượng bên trục y2, cùng màu với giá
        volume_colors = [
            'green' if row['close'] > row['open'] else 'red'
            for _, row in df.iterrows()
        ]
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=volume_colors
        ), row=1, col=1, secondary_y=True)

        # Biểu đồ MACD (được chuyển sang hàng 2)
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['MACD_Hist'],
            name='MACD Histogram'
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['MACD'],
            mode='lines',
            line=dict(color='green', width=1),
            name='MACD'
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['Signal'],
            mode='lines',
            line=dict(color='red', width=1),
            name='Signal'
        ), row=2, col=1)

        # Biểu đồ RSI (hàng 3)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['RSI'],
            mode='lines',
            line=dict(color='purple', width=1),
            name='RSI'
        ), row=3, col=1)

        # Đường giới hạn RSI
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=[70] * len(df),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Overbought (70)'
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=[30] * len(df),
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Oversold (30)'
        ), row=3, col=1)

        # Layout
        fig.update_layout(
            title=f"{symbol} Technical Analysis Chart (1H)",
            template="plotly_dark",
            height=1200,  # Tăng chiều cao biểu đồ tổng thể
            xaxis_rangeslider_visible=False
        )



        # Lưu biểu đồ thành HTML
        temp_file = f"{symbol.replace('/', '_')}_chart.html"
        fig.write_html(temp_file)


        # Gửi file HTML qua Telegram
        if update.callback_query:
            with open(temp_file, 'rb') as html_file:
                await update.callback_query.message.reply_document(document=html_file, filename=temp_file)
        else:
            with open(temp_file, 'rb') as html_file:
                await update.message.reply_document(document=html_file, filename=temp_file)


        # Xóa file tạm
        os.remove(temp_file)
    except Exception as e:
        if update.callback_query:
            await update.callback_query.message.reply_text(f"Đã xảy ra lỗi: {e}")
        else:
            await update.message.reply_text(f"Đã xảy ra lỗi: {e}")


async def top(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gửi danh sách top 10 cặp giao dịch tăng, giảm mạnh nhất và có khối lượng lớn nhất với nút tương tác."""
    try:
        # Lấy dữ liệu thị trường từ KuCoin
        markets = exchange.fetch_tickers()
        data = []
        volume_data = []

        # Tính toán phần trăm biến động giá và khối lượng giao dịch
        for symbol, ticker in markets.items():
            change = ticker.get('percentage')
            volume_24h = ticker.get('quoteVolume', 0)  # Khối lượng giao dịch 24 giờ
            if change is not None:
                data.append((symbol, change))
            if volume_24h > 0:
                volume_data.append((symbol, volume_24h))

        # Lấy top 10 tăng, giảm mạnh nhất và khối lượng lớn nhất
        top_gainers = sorted(data, key=lambda x: x[1], reverse=True)[:10]
        top_losers = sorted(data, key=lambda x: x[1])[:10]
        top_volumes = sorted(volume_data, key=lambda x: x[1], reverse=True)[:10]

        # Tạo danh sách nút tương tác cho top tăng
        gainers_keyboard = [
            [InlineKeyboardButton(f"{symbol}: +{change:.2f}%", callback_data=symbol)]
            for symbol, change in top_gainers
        ]

        # Tạo danh sách nút tương tác cho top giảm
        losers_keyboard = [
            [InlineKeyboardButton(f"{symbol}: {change:.2f}%", callback_data=symbol)]
            for symbol, change in top_losers
        ]

        # Tạo danh sách nút tương tác cho khối lượng giao dịch lớn nhất
        volumes_keyboard = [
            [InlineKeyboardButton(f"{symbol}: {volume:.2f} USD", callback_data=symbol)]
            for symbol, volume in top_volumes
        ]

        # Gửi danh sách top tăng mạnh nhất
        await update.message.reply_text(
            "Top 10 cặp giao dịch tăng mạnh nhất trong 24 giờ qua:",
            reply_markup=InlineKeyboardMarkup(gainers_keyboard)
        )

        # Gửi danh sách top giảm mạnh nhất
        await update.message.reply_text(
            "Top 10 cặp giao dịch giảm mạnh nhất trong 24 giờ qua:",
            reply_markup=InlineKeyboardMarkup(losers_keyboard)
        )

        # Gửi danh sách top khối lượng lớn nhất
        await update.message.reply_text(
            "Top 10 cặp giao dịch có khối lượng lớn nhất trong 24 giờ qua:",
            reply_markup=InlineKeyboardMarkup(volumes_keyboard)
        )
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")





async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý nút bấm từ danh sách /top để hiển thị biểu đồ kỹ thuật."""
    query = update.callback_query
    await query.answer()


    # Lấy mã giao dịch từ callback_data
    symbol = query.data
    context.chat_data["symbol"] = symbol  # Lưu vào chat_data để gọi lại nếu cần
    await chart(update, context)






async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Phân tích và gửi tín hiệu mua bán."""
    try:
        symbol = context.args[0] if context.args else None
        if not symbol:
            await update.message.reply_text("Vui lòng cung cấp mã giao dịch. Ví dụ: /signal BTC/USDT")
            return

        timeframe = '6h'
        limit = 500

        markets = exchange.load_markets()
        if symbol not in markets:
            await update.message.reply_text(f"Mã giao dịch không hợp lệ: {symbol}. Vui lòng kiểm tra lại.")
            return

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Chuyển đổi timestamp sang giờ Việt Nam
        df['timestamp'] = (
            pd.to_datetime(df['timestamp'], unit='ms')
            .dt.tz_localize('UTC')
            .dt.tz_convert(vietnam_tz)
        )

        # Tính toán các chỉ báo kỹ thuật
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA100'] = df['close'].rolling(window=100).mean()
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()

        # Phát hiện tín hiệu mua bán hiện tại
        last_row = df.iloc[-1]  # Lấy dòng dữ liệu cuối cùng
        signals_now = []

        # Thời điểm và giá hiện tại
        current_time = last_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        current_price = last_row['close']

        # Tín hiệu mua
        if last_row['close'] > last_row['MA50'] and last_row['MACD'] > last_row['Signal'] and last_row['RSI'] < 30:
            signals_now.append(f"\U0001F7E2 Mua: Giá {current_price:.2f} USD vào lúc {current_time}.")
        elif last_row['close'] <= last_row['BB_Lower']:
            signals_now.append(f"\U0001F7E2 Mua: Giá {current_price:.2f} USD vào lúc {current_time}.")

        # Tín hiệu bán
        if last_row['close'] < last_row['MA50'] and last_row['MACD'] < last_row['Signal'] and last_row['RSI'] > 70:
            signals_now.append(f"\U0001F534 Bán: Giá {current_price:.2f} USD vào lúc {current_time}.")
        elif last_row['close'] >= last_row['BB_Upper']:
            signals_now.append(f"\U0001F534 Bán: Giá {current_price:.2f} USD vào lúc {current_time}.")

        # Phát hiện tín hiệu mua bán trong 7 ngày qua
        signals_past = []
        for index, row in df.iterrows():
            if row['timestamp'] < (df['timestamp'].iloc[-1] - pd.Timedelta(days=7)):
                continue

            profit_margin = ((current_price - row['close']) / row['close']) * 100
            icon = "\U0001F7E2" if profit_margin > 0 else "\U0001F534"

            if row['close'] > row['MA50'] and row['MACD'] > row['Signal'] and row['RSI'] < 30:
                signals_past.append(f"\U0001F7E2 Mua: Giá {row['close']:.2f} USD vào lúc {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}. {icon} Lãi/Lỗ: {profit_margin:.2f}%")
            elif row['close'] <= row['BB_Lower']:
                signals_past.append(f"\U0001F7E2 Mua: Giá {row['close']:.2f} USD vào lúc {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}. {icon} Lãi/Lỗ: {profit_margin:.2f}%")

            if row['close'] < row['MA50'] and row['MACD'] < row['Signal'] and row['RSI'] > 70:
                signals_past.append(f"\U0001F534 Bán: Giá {row['close']:.2f} USD vào lúc {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}. {icon} Lãi/Lỗ: {profit_margin:.2f}%")
            elif row['close'] >= row['BB_Upper']:
                signals_past.append(f"\U0001F534 Bán: Giá {row['close']:.2f} USD vào lúc {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}. {icon} Lãi/Lỗ: {profit_margin:.2f}%")

        # Gửi tín hiệu qua Telegram
        signal_message = f"Tín hiệu giao dịch cho {symbol}:\n"
        if signals_now:
            signal_message += "\nTín hiệu hiện tại:\n" + "\n".join(signals_now)
        else:
            signal_message += "\nHiện tại không có tín hiệu rõ ràng."

        if signals_past:
            signal_message += "\n\nTín hiệu trong 7 ngày qua:\n" + "\n".join(signals_past)
        else:
            signal_message += "\n\nKhông có tín hiệu trong 7 ngày qua."

        await update.message.reply_text(signal_message)

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")



async def monitor_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Theo dõi tín hiệu và tự động gửi thông báo đến subscribers."""
    try:
        markets = exchange.load_markets()
        symbols = list(markets.keys())  # Danh sách mã giao dịch theo dõi
        timeframe = '1h'
        limit = 200

        for symbol in symbols:
            markets = exchange.load_markets()
            if symbol not in markets:
                continue

            # Lấy dữ liệu từ KuCoin
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Tính toán các chỉ báo kỹ thuật
            df['MA50'] = df['close'].rolling(window=50).mean()
            df['EMA12'] = df['close'].ewm(span=12).mean()
            df['EMA26'] = df['close'].ewm(span=26).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['Signal'] = df['MACD'].ewm(span=9).mean()
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['BB_Middle'] = df['close'].rolling(window=20).mean()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()

            # Phát hiện tín hiệu mua bán
            last_row = df.iloc[-1]
            signals = []

            # Thời điểm và giá hiện tại
            current_time = last_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            current_price = last_row['close']

            # Tín hiệu mua
            if last_row['close'] > last_row['MA50'] and last_row['MACD'] > last_row['Signal'] and last_row['RSI'] < 30:
                signals.append(f"Mua: Giá {current_price:.2f} USD tại {current_time}. Lý do: Cắt MA50, MACD tăng và RSI dưới 30.")
            elif last_row['close'] <= last_row['BB_Lower']:
                signals.append(f"Mua: Giá {current_price:.2f} USD tại {current_time}. Lý do: Gần dải BB dưới (quá bán).")

            # Tín hiệu bán
            if last_row['close'] < last_row['MA50'] and last_row['MACD'] < last_row['Signal'] and last_row['RSI'] > 70:
                signals.append(f"Bán: Giá {current_price:.2f} USD tại {current_time}. Lý do: Cắt MA50, MACD giảm và RSI trên 70.")
            elif last_row['close'] >= last_row['BB_Upper']:
                signals.append(f"Bán: Giá {current_price:.2f} USD tại {current_time}. Lý do: Gần dải BB trên (quá mua).")

            # Gửi tín hiệu cho tất cả subscribers
            if signals:
                signal_message = f"Tín hiệu tự động cho {symbol}:\n" + "\n".join(signals)
                for chat_id in subscribers:
                    await context.bot.send_message(chat_id=chat_id, text=signal_message)
    except Exception as e:
        print(f"Error in monitor_signals: {e}")



async def start_monitoring(application: Application):
    """Khởi tạo job để theo dõi tín hiệu giao dịch."""
    job_queue = application.job_queue
    job_queue.run_repeating(monitor_signals, interval=300, first=10)  # Chạy mỗi 5 phút

async def set_webhook(application: Application):
    """Thiết lập Webhook."""
    await application.bot.set_webhook(WEBHOOK_URL)

def main():
    # Lấy cổng từ biến môi trường hoặc sử dụng cổng mặc định
    port = int(os.getenv("PORT", 8080))
    print(f"Đang sử dụng cổng: {port}")  # Log kiểm tra cổng

    # Khởi tạo ứng dụng Telegram bot
    application = Application.builder().token(TOKEN).build()

    # Đăng ký các handler
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("subscribe", subscribe))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe))
    application.add_handler(CommandHandler("chart", chart))
    application.add_handler(CommandHandler("signal", signal))
    application.add_handler(CommandHandler("top", top))  # Thêm handler cho /top
    application.add_handler(CommandHandler("cap", current_price))  # Thêm handler cho /cap
    application.add_handler(CallbackQueryHandler(button))  # Thêm handler cho nút bấm từ /top

    # Chạy webhook
    application.run_webhook(
        listen="0.0.0.0",
        port=port,
        webhook_url=WEBHOOK_URL
    )

if __name__ == "__main__":
    main()





